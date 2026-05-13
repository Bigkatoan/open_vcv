"""Tests for src/trainers/logger.py"""
import os
import csv
import tempfile
import pytest
from src.trainers.logger import Logger


@pytest.fixture
def logger(tmp_path):
    lg = Logger(str(tmp_path))
    yield lg
    lg.close()


def test_files_created(tmp_path):
    lg = Logger(str(tmp_path))
    lg.close()
    for fname in ('train.log', 'losses.csv', 'metrics.csv',
                  'iter_losses.csv', 'probe_results.csv'):
        assert (tmp_path / fname).exists(), f"missing {fname}"


def test_log_probe_writes_csv(logger, tmp_path):
    result = {'top1': 72.5, 'top5': 91.3, 'n_train': 50000, 'n_test': 10000, 'time_s': 85.0}
    logger.log_probe(epoch=10, result=result, dataset='cifar10', probe_epochs=20)

    rows = list(csv.DictReader(open(tmp_path / 'probe_results.csv')))
    assert len(rows) == 1
    assert rows[0]['epoch']   == '10'
    assert rows[0]['dataset'] == 'cifar10'
    assert float(rows[0]['top1']) == pytest.approx(72.5)
    assert float(rows[0]['top5']) == pytest.approx(91.3)


def test_log_probe_multiple_epochs(logger, tmp_path):
    result = {'top1': 60.0, 'top5': 85.0, 'n_train': 50000, 'n_test': 10000, 'time_s': 80.0}
    logger.log_probe(5, result, 'cifar10', 20)
    logger.log_probe(10, {**result, 'top1': 65.0}, 'cifar10', 20)
    rows = list(csv.DictReader(open(tmp_path / 'probe_results.csv')))
    assert len(rows) == 2
    assert rows[1]['epoch'] == '10'


def test_log_probe_writes_to_log(logger, tmp_path):
    result = {'top1': 72.5, 'top5': 91.3, 'n_train': 50000, 'n_test': 10000, 'time_s': 85.0}
    logger.log_probe(10, result, 'cifar10', 20)
    logger.close()
    log_text = (tmp_path / 'train.log').read_text()
    assert 'LinearProbe' in log_text
    assert '72.50%' in log_text


def test_info(logger, tmp_path):
    logger.info("hello test")
    logger.close()
    assert "hello test" in (tmp_path / 'train.log').read_text()
