"""Tests du verrou d'instance unique — la protection contre le conflit micro."""

import os

import pytest

from audio_source import AlreadyRunning, acquire_single_instance_lock


def test_une_seconde_instance_est_refusee(tmp_path):
    path = str(tmp_path / "babyphone.lock")
    first = acquire_single_instance_lock(path)
    with pytest.raises(AlreadyRunning):
        acquire_single_instance_lock(path)
    first.close()


def test_le_verrou_est_repris_apres_liberation(tmp_path):
    path = str(tmp_path / "babyphone.lock")
    first = acquire_single_instance_lock(path)
    first.close()  # simule la mort du processus : le noyau libère le verrou
    second = acquire_single_instance_lock(path)  # ne doit pas lever
    second.close()


def test_le_verrou_contient_le_pid(tmp_path):
    path = str(tmp_path / "babyphone.lock")
    handle = acquire_single_instance_lock(path)
    assert open(path).read().strip() == str(os.getpid())
    handle.close()
