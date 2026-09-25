"""Offline regressions: deposited numbers, schedule selection, shell exit propagation.

No DFT, MLIP inference, training, checkpoint download or dependency installation.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'mlip/r23_2026-09'


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        scratch = REPO / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='offline-repro-', dir=scratch)
        self.work = Path(self.temp.name)
        self.env = dict(os.environ, R23_OUT=str(self.work), R23_RUNS=str(self.work/'runs'),
                        PYTHONIOENCODING='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def run_python(self, script, *args):
        return subprocess.run([sys.executable, str(REPO/'scripts'/script), *args],
                              env=self.env, capture_output=True, text=True, encoding='utf-8')

    def bundle(self, hold):
        return json.loads((DATA/f'evaluations_{hold}.json').read_text())

    def aggregate(self, hold, bundle):
        for tag, ev in bundle.items():
            (self.work/f'eval_{tag}.json').write_text(json.dumps(ev))
        result = self.run_python('aggregate2.py', hold)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((self.work/f'ladder_{hold}.json').read_text())

    def test_deposited_seed_statistics_both_holdouts(self):
        for hold in ['mackinawite', 'marcasite']:
            with self.subTest(hold=hold):
                new = self.aggregate(hold, self.bundle(hold))
                old = json.loads((DATA/f'ladder_{hold}.json').read_text())
                for kind, n in [('sp', 8), ('neb', 3)]:
                    self.assertEqual(new['discriminator'][kind]['n_pairs'], n)
                    for key in ['seeds', 'delta_per_seed', 'delta_mean', 'delta_sd', 'ci95', 'p_exact']:
                        self.assertEqual(new['discriminator'][kind][key], old['discriminator'][kind][key])
                self.assertIn('gate failed or is undocumented', new['discriminator']['sp']['verdict'])

    def test_missing_neb_in_one_rung_excludes_pair(self):
        bundle = self.bundle('mackinawite')
        del bundle['S3_s3']['neb_selfconsistent']
        new = self.aggregate('mackinawite', bundle)
        self.assertEqual(new['discriminator']['neb']['seeds'], [1, 2])
        self.assertEqual(new['discriminator']['sp']['n_pairs'], 8)

    def test_no_neb_does_not_fall_back_to_single_points(self):
        bundle = self.bundle('mackinawite')
        for ev in bundle.values():
            ev.pop('neb_selfconsistent', None)
        new = self.aggregate('mackinawite', bundle)
        self.assertNotIn('neb', new['discriminator'])

    def schedules(self):
        sweep = json.loads((DATA/'schedule_sweep.json').read_text())
        (self.work/'tune_v2.json').write_text(json.dumps(sweep['results']))
        with tarfile.open(DATA/'training_logs.tar.gz') as archive:
            for tag in sweep['configurations']:
                name = f'mackinawite/{tag}.log'
                target = self.work/'runs'/tag/'train.log'
                target.parent.mkdir(parents=True)
                target.write_bytes(archive.extractfile(name).read())

    def test_all_five_schedules_select_long_control(self):
        self.schedules()
        result = self.run_python('pick_schedule.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        chosen = json.loads((self.work/'chosen.json').read_text())
        self.assertEqual(chosen['tag'], 'tD')
        self.assertEqual(chosen['epochs'], 4000)
        self.assertTrue(chosen['gate_passed'])
        self.assertAlmostEqual(chosen['valid_force_rmse_meV_A'], 12.08, places=2)
        # Selection must follow validation RMSE, not a hardcoded preference for tD.
        (self.work/'runs/tE/train.log').write_text('Epoch 4000: RMSE_F= 1.0 meV/A\n')
        self.assertEqual(self.run_python('pick_schedule.py').returncode, 0)
        self.assertEqual(json.loads((self.work/'chosen.json').read_text())['tag'], 'tE')

    def test_missing_passing_validation_log_prevents_selection(self):
        self.schedules()
        (self.work/'runs/tD/train.log').unlink()
        result = self.run_python('pick_schedule.py')
        self.assertNotEqual(result.returncode, 0)
        chosen = json.loads((self.work/'chosen.json').read_text())
        self.assertFalse(chosen['gate_passed'])
        self.assertEqual(chosen['status'], 'invalid')

    def test_failed_selection_invalidates_previous_passing_choice(self):
        self.schedules()
        (self.work/'chosen.json').write_text(json.dumps({'tag':'tD', 'gate_passed':True,
                                                       'sentinel':'previous_run'}))
        (self.work/'runs/tD/train.log').unlink()
        result = self.run_python('pick_schedule.py')
        self.assertNotEqual(result.returncode, 0)
        chosen = json.loads((self.work/'chosen.json').read_text())
        self.assertEqual(chosen['status'], 'invalid')
        self.assertFalse(chosen['gate_passed'])
        self.assertIsNone(chosen['tag'])
        self.assertIn('Missing validation', chosen['source_error'])
        self.assertNotIn('sentinel', chosen)

    def test_missing_or_partial_selftest_never_passes(self):
        for defect in ['absent', 'empty', 'partial']:
            with self.subTest(defect=defect):
                bundle = self.bundle('mackinawite')
                if defect == 'absent':
                    bundle['S0'].pop('P1_selftest')
                elif defect == 'empty':
                    bundle['S0']['P1_selftest'] = {}
                else:
                    bundle['S0']['P1_selftest'].pop('marcasite')
                for tag, ev in bundle.items():
                    (self.work/f'eval_{tag}.json').write_text(json.dumps(ev))
                result = self.run_python('aggregate2.py', 'mackinawite')
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('SELF-TEST G1 on DFT: PASS', result.stdout)
                self.assertFalse((self.work/'ladder_mackinawite.json').exists())

    def test_malformed_profile_cannot_produce_partial_success(self):
        inputs = self.work/'profiles'
        inputs.mkdir()
        for source in (REPO/'mlip/r22_2026-09').glob('*.json'):
            shutil.copyfile(source, inputs/source.name)
        path = inputs/'band_mackinawite.json'
        data = json.loads(path.read_text())
        data['mace:large']['profile_meV'].pop()
        path.write_text(json.dumps(data))
        target = self.work/'per_image_stats.json'
        self.env.update(MLIP_RUN=str(inputs), MLIP_STATS_OUTPUT=str(target))
        result = self.run_python('per_image_stats.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid profile length', result.stderr)
        self.assertFalse(target.exists())

    def test_training_exit_status_without_running_training(self):
        bash = os.environ.get('BASH_EXECUTABLE')
        if not bash and os.name == 'nt':
            candidate = Path('C:/Program Files/Git/bin/bash.exe')
            bash = str(candidate) if candidate.exists() else None
        bash = bash or shutil.which('bash')
        if not bash:
            self.skipTest('Bash is required for the shell-wrapper regression')
        fake = self.work/'fake-python'
        fake.write_text('#!/bin/sh\nif [ "$1" = "-m" ]; then exit 7; fi\nexit 0\n', newline='\n')
        fake.chmod(0o755)
        env = dict(self.env, R23_WORK=self.work.as_posix(), R23_PYTHON=fake.as_posix(),
                   R23_OUT=self.work.as_posix(), R23_RUNS=(self.work/'runs').as_posix())
        result = subprocess.run([bash, (REPO/'scripts/train4.sh').as_posix(),
                                 'S1', '1', '200', '10000', '100', '.001', 'float64', 'mock'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)

    def test_profile_statistics_use_checkout_data(self):
        self.env['MLIP_STATS_OUTPUT'] = str(self.work/'per_image_stats.json')
        result = self.run_python('per_image_stats.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.work/'per_image_stats.json').read_text())
        self.assertEqual(len(report), 5)
        self.assertEqual(sum(len(v['models']) for v in report.values()), 45)
        self.assertEqual(report['marcasite']['excluded_images'], [1])
        self.env['MLIP_RUN'] = str(self.work/'missing-profiles')
        result = self.run_python('per_image_stats.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Expected nine model profiles', result.stderr)


if __name__ == '__main__':
    unittest.main()
