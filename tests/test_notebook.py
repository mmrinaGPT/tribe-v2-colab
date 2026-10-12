"""Offline regressions for the real notebook's transcription boundary."""
import ast
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
import contextlib
import io
from unittest.mock import patch

import pandas as pd
import numpy as np

NOTEBOOK = Path(__file__).resolve().parents[1] / 'TRIBE_v2_WhisperX_fix.ipynb'

class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        source = ''.join(self.nb['cells'][6]['source'])
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = dict(Path=Path, pd=pd, math=math, json=json,
                        tempfile=tempfile, os=__import__('os'),
                        subprocess=__import__('subprocess'),
                        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
                        CACHE_FOLDER=Path(self.tmp.name), NLTK_DIR=str(Path(self.tmp.name) / 'nltk_data'),
                        WHISPERX_VERSION='3.8.6')
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), self.env)
        self.calls = []

    def run_transcript(self, words, language='english', error=False, json_output=True):
        def run(cmd, **kwargs):
            self.calls.append((cmd, kwargs))
            directory = Path(cmd[cmd.index('--output_dir') + 1])
            if json_output:
                (directory / 'test.json').write_text(json.dumps({'segments': [
                    {'text': 'Hello world', 'words': words}]}), encoding='utf-8')
            return types.SimpleNamespace(returncode=1 if error else 0,
                                         stdout='stdout marker', stderr='stderr marker')
        with patch('subprocess.run', side_effect=run):
            return self.env['_patched_transcript'](Path('test.wav'), language)

    def test_all_cells_compile(self):
        self.assertEqual(self.nb['nbformat'], 4)
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(NOTEBOOK), 'exec')

    def test_gpu_and_shared_nltk(self):
        result = self.run_transcript([{'word': 'Hello', 'start': 0.0, 'end': 0.5}])
        self.assertEqual(result.iloc[0]['duration'], 0.5)
        cmd, kwargs = self.calls[0]
        self.assertIn('whisperx==3.8.6', cmd)
        self.assertEqual(cmd[cmd.index('--compute_type') + 1], 'float16')
        self.assertEqual(kwargs['env']['NLTK_DATA'], self.env['NLTK_DIR'])
        self.assertNotIn('MPLBACKEND', kwargs['env'])
        self.assertNotIn('--no_align', cmd)

    def test_cpu_and_non_english(self):
        self.env['torch'].cuda.is_available = lambda: False
        self.run_transcript([{'word': 'Bonjour', 'start': 0, 'end': 1}], language='french')
        cmd = self.calls[0][0]
        self.assertEqual(cmd[cmd.index('--compute_type') + 1], 'int8')
        self.assertEqual(cmd[cmd.index('--device') + 1], 'cpu')
        self.assertNotIn('--align_model', cmd)

    def test_invalid_timings_are_excluded(self):
        result = self.run_transcript([
            {'word': 'valid', 'start': 0, 'end': 1},
            {'word': 'missing', 'start': 1},
            {'word': 'backwards', 'start': 2, 'end': 1},
            {'word': 'infinite', 'start': 0, 'end': float('inf')},
            {'word': 'null', 'start': None, 'end': 1},
        ])
        self.assertEqual(result['text'].tolist(), ['valid'])

    def test_no_aligned_words_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'No aligned words'):
            self.run_transcript([])

    def test_full_error_log_survives(self):
        with self.assertRaisesRegex(RuntimeError, 'stdout marker.*|WhisperX exited'):
            self.run_transcript([], error=True)
        log = (self.env['CACHE_FOLDER'] / 'whisperx_last.log').read_text()
        self.assertIn('stdout marker', log)
        self.assertIn('stderr marker', log)

    def test_missing_json_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, 'produced no JSON'):
            self.run_transcript([], json_output=False)

    def test_unsupported_language_fails_before_subprocess(self):
        with self.assertRaises(ValueError):
            self.run_transcript([], language='unsupported')
        self.assertEqual(self.calls, [])

    def test_restart_gate_before_numpy_import(self):
        source = ''.join(self.nb['cells'][4]['source'])
        with self.assertRaisesRegex(RuntimeError, 'Restart the session'):
            exec(source, {'_TRIBE_RESTART_REQUIRED': True})

    def _audio_prediction_env(self, fail=False):
        data = types.SimpleNamespace(features_to_use=['text', 'audio', 'video'],
                                     num_workers=20, batch_size=8)
        def predict(events):
            self.assertEqual(data.features_to_use, ['audio'])
            self.assertEqual(events.type.tolist(), ['Audio'])
            self.assertEqual(data.num_workers, 0)
            if fail:
                raise RuntimeError('inference failed')
            return np.ones((2, 3), dtype=np.float32), [object(), object()]
        model = types.SimpleNamespace(data=data, predict=predict)
        env = dict(model=model, np=np, CACHE_FOLDER=Path(self.tmp.name),
                   df=pd.DataFrame({'type': ['Audio', 'Word'], 'duration': [1., 1.]}))
        return model, env

    def test_audio_prediction_excludes_words_and_restores_settings(self):
        model, env = self._audio_prediction_env()
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(self.nb['cells'][9]['source']), env)
        self.assertEqual(model.data.features_to_use, ['text', 'audio', 'video'])
        self.assertEqual(model.data.num_workers, 20)
        self.assertEqual(model.data.batch_size, 8)
        saved = np.load(Path(self.tmp.name) / 'audio_only_brain_activity.npy', allow_pickle=False)
        self.assertEqual(saved.shape, (2, 3))

    def test_audio_prediction_failure_restores_settings(self):
        model, env = self._audio_prediction_env(fail=True)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                exec(''.join(self.nb['cells'][9]['source']), env)
        self.assertEqual(model.data.features_to_use, ['text', 'audio', 'video'])
        self.assertEqual(model.data.num_workers, 20)
        self.assertEqual(model.data.batch_size, 8)

if __name__ == '__main__':
    unittest.main()
