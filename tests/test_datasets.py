"""
PerceptOS dataset manager and model registry tests.

The two tests that carry the most weight are about refusals:

* :meth:`TestSplitting.test_a_random_split_is_refused_without_acknowledgement`
* :meth:`TestEvaluation.test_a_majority_class_predictor_is_exposed`

The first prevents the leak that makes rover-imagery accuracy meaningless. The
second prevents the model that scores 87% by predicting "soil" everywhere and
never once identifying a hazard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.datasets.manager import (
    DatasetManager,
    DatasetVersion,
    LeakageError,
    Sample,
    detect_leakage,
)
from perceptos.datasets.registry import (
    ModelRegistry,
    confusion_matrix,
)


def make_samples(groups=("sol001", "sol002", "sol003"), per_group=6,
                 labelled=True):
    """Samples that mimic rover imagery: several frames per sol."""
    samples = []
    for group in groups:
        for i in range(per_group):
            samples.append(Sample(
                id=f"{group}_{i:03d}",
                path=f"/data/{group}/{i:03d}.png",
                sha256=f"{abs(hash((group, i))):064x}"[:64],
                label_path=f"/labels/{group}_{i:03d}.png" if labelled else None,
                group=group,
            ))
    return samples


# ===========================================================================
# Content-addressed versioning
# ===========================================================================

class TestDatasetVersion:
    def test_identical_content_gives_an_identical_id(self):
        """A version id names bytes; 'v3' names nothing."""
        a = DatasetVersion(name="terrain", samples=make_samples())
        b = DatasetVersion(name="terrain", samples=make_samples())
        assert a.version_id == b.version_id

    def test_sample_order_does_not_change_the_id(self):
        samples = make_samples()
        a = DatasetVersion(name="terrain", samples=samples)
        b = DatasetVersion(name="terrain", samples=list(reversed(samples)))
        assert a.version_id == b.version_id

    def test_changing_one_sample_changes_the_id(self):
        samples = make_samples()
        a = DatasetVersion(name="terrain", samples=samples)
        altered = list(samples)
        altered[0] = Sample(id=altered[0].id, path=altered[0].path,
                            sha256="0" * 64, group=altered[0].group)
        assert DatasetVersion(name="terrain", samples=altered).version_id != a.version_id

    def test_statistics_report_grouping_and_label_coverage(self):
        version = DatasetVersion(name="t", samples=make_samples())
        stats = version.statistics()
        assert stats["samples"] == 18
        assert stats["groups"] == 3
        assert stats["label_coverage"] == 1.0
        assert stats["ungrouped"] == 0

    def test_partial_labelling_is_reported_honestly(self):
        samples = make_samples() + make_samples(("sol009",), labelled=False)
        version = DatasetVersion(name="t", samples=samples)
        assert 0.0 < version.label_coverage() < 1.0

    def test_an_edited_manifest_is_detected_on_load(self):
        """A model trained against the recorded id no longer names this data."""
        version = DatasetVersion(name="t", samples=make_samples())
        manifest = version.to_manifest()
        manifest["samples"][0]["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="has been edited"):
            DatasetVersion.from_manifest(manifest)

    def test_an_unedited_manifest_round_trips(self):
        version = DatasetVersion(name="t", samples=make_samples())
        restored = DatasetVersion.from_manifest(version.to_manifest())
        assert restored.version_id == version.version_id
        assert len(restored.samples) == len(version.samples)


# ===========================================================================
# Splitting and leakage
# ===========================================================================

class TestSplitting:
    def test_a_random_split_is_refused_without_acknowledgement(self):
        """Random splitting of sequential frames measures memorisation."""
        version = DatasetVersion(name="t", samples=make_samples())
        with pytest.raises(LeakageError, match="near-duplicate"):
            version.split(group_key=None)

    def test_a_random_split_is_allowed_when_asserted(self):
        version = DatasetVersion(name="t", samples=make_samples())
        result = version.split(group_key=None, allow_random=True)
        assert "independence asserted" in result.strategy

    def test_a_random_split_actually_leaks_groups(self):
        """Demonstrates the harm the default guards against."""
        version = DatasetVersion(name="t", samples=make_samples(per_group=20))
        result = version.split(group_key=None, allow_random=True, seed=1)
        assert result.leakage.shared_groups, (
            "a random split over sequential frames should put the same sol in "
            "more than one partition"
        )

    def test_a_grouped_split_keeps_each_sol_whole(self):
        version = DatasetVersion(
            name="t", samples=make_samples(tuple(f"sol{i:03d}" for i in range(20))))
        result = version.split(group_key="group", seed=0)
        assert result.leakage.clean
        assert not result.leakage.shared_groups

        train = {s.group for s in result.train}
        test = {s.group for s in result.test}
        assert not (train & test)

    def test_ungrouped_samples_block_a_grouped_split(self):
        samples = make_samples() + [Sample(id="x", path="/x.png",
                                           sha256="a" * 64, group=None)]
        version = DatasetVersion(name="t", samples=samples)
        with pytest.raises(LeakageError, match="no group key"):
            version.split(group_key="group")

    def test_fractions_must_sum_to_one(self):
        version = DatasetVersion(name="t", samples=make_samples())
        with pytest.raises(ValueError, match="sum to 1.0"):
            version.split(train=0.9, validation=0.2, test=0.2)

    def test_an_empty_dataset_cannot_be_split(self):
        with pytest.raises(ValueError, match="empty dataset"):
            DatasetVersion(name="t", samples=[]).split()

    def test_the_split_is_deterministic_for_a_given_seed(self):
        version = DatasetVersion(
            name="t", samples=make_samples(tuple(f"sol{i:03d}" for i in range(20))))
        a = version.split(seed=7)
        b = version.split(seed=7)
        assert [s.id for s in a.test] == [s.id for s in b.test]

    def test_too_few_groups_to_split_is_refused(self):
        """An empty partition yields metrics that look like results."""
        version = DatasetVersion(name="t", samples=make_samples(
            ("sol001", "sol002", "sol003", "sol004")))
        with pytest.raises(ValueError, match="came out empty"):
            version.split(group_key="group", seed=0)

    def test_the_refusal_names_the_real_cause(self):
        version = DatasetVersion(name="t", samples=make_samples(("a", "b", "c")))
        with pytest.raises(ValueError, match="Add more group"):
            version.split(group_key="group")

    def test_enough_groups_splits_into_three_non_empty_partitions(self):
        version = DatasetVersion(
            name="t", samples=make_samples(tuple(f"sol{i:03d}" for i in range(20))))
        result = version.split(group_key="group", seed=0)
        assert all(size > 0 for size in result.sizes.values())

    def test_different_seeds_give_different_splits(self):
        version = DatasetVersion(
            name="t", samples=make_samples(tuple(f"sol{i:03d}" for i in range(20))))
        assert ([s.id for s in version.split(seed=1).test]
                != [s.id for s in version.split(seed=2).test])


class TestLeakageDetection:
    def test_a_clean_split_is_reported_clean(self):
        train = make_samples(("sol001", "sol002"))
        test = make_samples(("sol003",))
        report = detect_leakage(train, [], test)
        assert report.clean
        assert "No leakage" in report.note

    def test_a_shared_group_is_caught(self):
        shared = make_samples(("sol001",), per_group=10)
        report = detect_leakage(shared[:5], [], shared[5:])
        assert not report.clean
        assert "sol001" in report.shared_groups
        assert "overstates real performance" in report.note

    def test_byte_identical_samples_are_caught(self):
        """Duplicated files under different names are a real dataset defect."""
        a = Sample(id="a", path="/a.png", sha256="d" * 64, group="sol001")
        b = Sample(id="b", path="/b.png", sha256="d" * 64, group="sol002")
        report = detect_leakage([a], [], [b])
        assert not report.clean
        assert "d" * 64 in report.duplicate_hashes

    def test_a_repeated_id_is_caught(self):
        sample = Sample(id="same", path="/a.png", sha256="a" * 64, group="s1")
        other = Sample(id="same", path="/b.png", sha256="b" * 64, group="s2")
        report = detect_leakage([sample], [], [other])
        assert not report.clean
        assert "same" in report.shared_ids

    def test_ungrouped_data_is_flagged_as_uncheckable(self):
        """Absence of evidence, stated as such."""
        a = Sample(id="a", path="/a.png", sha256="a" * 64, group=None)
        b = Sample(id="b", path="/b.png", sha256="b" * 64, group=None)
        report = detect_leakage([a], [], [b])
        assert report.clean
        assert "could not be checked" in report.note


class TestDatasetManager:
    def test_save_and_load_round_trip(self, tmp_path):
        manager = DatasetManager(tmp_path)
        version = manager.create("terrain", make_samples())
        manager.save(version)
        assert manager.load(version.version_id).version_id == version.version_id

    def test_an_unknown_version_lists_what_exists(self, tmp_path):
        manager = DatasetManager(tmp_path)
        manager.save(manager.create("terrain", make_samples()))
        with pytest.raises(KeyError, match="Available"):
            manager.load("deadbeef")

    def test_ingest_groups_by_containing_directory(self, tmp_path):
        """A sol00203/ layout groups exactly right for rover imagery."""
        for sol in ("sol00203", "sol00204"):
            folder = tmp_path / "raw" / sol
            folder.mkdir(parents=True)
            for i in range(3):
                (folder / f"img{i}.png").write_bytes(f"{sol}{i}".encode())

        version = DatasetManager(tmp_path / "versions").ingest_directory(
            tmp_path / "raw", name="perseverance")
        assert len(version.samples) == 6
        assert set(version.group_counts()) == {"sol00203", "sol00204"}
        # Ids include the sol, so identically-named files across sols do not
        # collide and get reported as duplicates in two partitions.
        assert len({s.id for s in version.samples}) == 6

    def test_an_empty_directory_is_reported(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no files matching"):
            DatasetManager(tmp_path).ingest_directory(tmp_path / "empty", name="x")

    def test_hashes_come_from_file_contents(self, tmp_path):
        same_a = tmp_path / "a.png"
        same_b = tmp_path / "b.png"
        same_a.write_bytes(b"identical")
        same_b.write_bytes(b"identical")
        assert (Sample.from_file(same_a).sha256
                == Sample.from_file(same_b).sha256)


# ===========================================================================
# Evaluation
# ===========================================================================

CLASSES = ["soil", "bedrock", "sand", "big_rock"]


class TestEvaluation:
    def test_a_perfect_prediction_scores_one(self):
        truth = np.array([[0, 1], [2, 3]])
        result = ModelRegistry.evaluate([truth], [truth.copy()], CLASSES)
        assert result.pixel_accuracy == pytest.approx(1.0)
        assert result.mean_iou == pytest.approx(1.0)

    def test_unlabelled_pixels_are_excluded(self):
        """Counting AI4Mars's 255 as a class inflates every metric."""
        truth = np.array([[0, 255], [255, 1]])
        prediction = np.array([[0, 3], [3, 1]])
        result = ModelRegistry.evaluate([truth], [prediction], CLASSES)
        assert result.evaluated_pixels == 2
        assert result.pixel_accuracy == pytest.approx(1.0)

    def test_a_majority_class_predictor_is_exposed(self):
        """87% accuracy while never finding a single hazard."""
        truth = np.zeros((10, 10), dtype=np.uint8)
        truth[:1, :3] = 3                       # 3% big rock
        prediction = np.zeros((10, 10), dtype=np.uint8)   # predicts soil only

        result = ModelRegistry.evaluate([truth], [prediction], CLASSES)
        assert result.pixel_accuracy > 0.95
        assert result.beats_baseline is False
        assert result.per_class["big_rock"]["recall"] == 0.0
        assert result.weakest_class() == "big_rock"

        warnings = " ".join(result.warnings())
        assert "majority-class baseline" in warnings
        assert "unsafe" in warnings

    def test_an_absent_class_is_undefined_not_zero(self):
        """Reporting 0.0 implies failure at something never asked for."""
        truth = np.zeros((4, 4), dtype=np.uint8)
        result = ModelRegistry.evaluate([truth], [truth.copy()], CLASSES)
        assert result.per_class["big_rock"]["defined"] is False
        assert result.support["big_rock"] == 0
        assert result.mean_iou == pytest.approx(1.0), (
            "absent classes must not drag the mean down"
        )
        assert any("does not appear" in w for w in result.warnings())

    def test_class_imbalance_is_called_out(self):
        truth = np.zeros((10, 10), dtype=np.uint8)
        truth[0, 0] = 1
        result = ModelRegistry.evaluate([truth], [truth.copy()], CLASSES)
        assert any("covers 99%" in w or "dominated by it" in w
                   for w in result.warnings())

    def test_precision_and_recall_are_distinguished(self):
        truth = np.array([[3, 3, 0, 0]])
        prediction = np.array([[3, 0, 0, 0]])       # finds half the rock
        result = ModelRegistry.evaluate([truth], [prediction], CLASSES)
        assert result.per_class["big_rock"]["recall"] == pytest.approx(0.5)
        assert result.per_class["big_rock"]["precision"] == pytest.approx(1.0)

    def test_mismatched_counts_are_rejected(self):
        with pytest.raises(ValueError, match="predictions"):
            ModelRegistry.evaluate([np.zeros((2, 2))], [], CLASSES)

    def test_empty_class_names_are_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelRegistry.evaluate([np.zeros((2, 2))], [np.zeros((2, 2))], [])

    def test_confusion_matrix_shape_and_content(self):
        truth = np.array([0, 1, 1, 2])
        prediction = np.array([0, 1, 2, 2])
        matrix = confusion_matrix(truth, prediction, 3, ignore_value=None)
        assert matrix.shape == (3, 3)
        assert matrix[1, 1] == 1 and matrix[1, 2] == 1


# ===========================================================================
# Model registry
# ===========================================================================

class TestModelRegistry:
    def test_a_model_must_name_its_dataset_version(self):
        """Without it, provenance is unverifiable and metrics unreproducible."""
        with pytest.raises(ValueError, match="dataset version"):
            ModelRegistry().register(name="m", task="segmentation",
                                     dataset_version="", class_names=CLASSES)

    def test_registration_round_trips(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        record = registry.register(name="terrain-net", task="segmentation",
                                   dataset_version="a3f9c2e18b04",
                                   class_names=CLASSES, framework="pytorch")
        assert registry.get(record.id).dataset_version == "a3f9c2e18b04"
        assert len(registry.list_models()) == 1

    def test_a_leaked_split_is_stamped_on_the_model_card(self, tmp_path):
        record = ModelRegistry(tmp_path).register(
            name="leaky", task="segmentation", dataset_version="abc123",
            class_names=CLASSES, split_leakage_clean=False,
            split_strategy="random")
        card = record.model_card()
        assert "LEAKED SPLIT" in card
        assert "must not be used to justify deployment" in card

    def test_a_clean_split_is_recorded_as_such(self, tmp_path):
        record = ModelRegistry(tmp_path).register(
            name="clean", task="segmentation", dataset_version="abc123",
            class_names=CLASSES, split_leakage_clean=True)
        assert "clean" in record.model_card()
        assert "LEAKED" not in record.model_card()

    def test_an_unaudited_split_says_so(self, tmp_path):
        record = ModelRegistry(tmp_path).register(
            name="unknown", task="segmentation", dataset_version="abc123",
            class_names=CLASSES)
        assert "not performed" in record.model_card()

    def test_the_model_card_carries_per_class_metrics(self, tmp_path):
        truth = np.array([[0, 1], [2, 3]])
        evaluation = ModelRegistry.evaluate([truth], [truth.copy()], CLASSES,
                                            dataset_version="abc123")
        record = ModelRegistry(tmp_path).register(
            name="m", task="segmentation", dataset_version="abc123",
            class_names=CLASSES, metrics={"evaluation": evaluation.to_dict()})
        card = record.model_card()
        assert "| Class | IoU | Precision | Recall |" in card
        assert "Majority-class baseline" in card
        assert "predicting only the most common class" in card

    def test_models_on_different_versions_are_not_comparable(self, tmp_path):
        """Putting their numbers side by side would imply that they are."""
        registry = ModelRegistry(tmp_path)
        a = registry.register(name="a", task="segmentation",
                              dataset_version="v_one", class_names=CLASSES)
        b = registry.register(name="b", task="segmentation",
                              dataset_version="v_two", class_names=CLASSES)
        comparison = registry.compare([a.id, b.id])
        assert comparison["comparable"] is False
        assert "not directly comparable" in comparison["note"]

    def test_models_on_one_version_are_comparable(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        a = registry.register(name="a", task="segmentation",
                              dataset_version="same", class_names=CLASSES)
        b = registry.register(name="b", task="segmentation",
                              dataset_version="same", class_names=CLASSES)
        assert registry.compare([a.id, b.id])["comparable"] is True

    def test_an_unknown_model_is_a_key_error(self, tmp_path):
        with pytest.raises(KeyError, match="no model"):
            ModelRegistry(tmp_path).get("nope")


# ===========================================================================
# End to end
# ===========================================================================

class TestDatasetToModelProvenance:
    def test_a_model_traces_back_to_exact_bytes(self, tmp_path):
        """The chain the whole module exists to support."""
        manager = DatasetManager(tmp_path / "versions")
        version = manager.create(
            "terrain", make_samples(tuple(f"sol{i:03d}" for i in range(20))))
        manager.save(version)

        split = version.split(group_key="group", seed=3)
        assert split.leakage.clean

        registry = ModelRegistry(tmp_path / "models")
        record = registry.register(
            name="terrain-net", task="segmentation",
            dataset_version=version.version_id, class_names=CLASSES,
            split_strategy=split.strategy,
            split_leakage_clean=split.leakage.clean,
        )

        # From the model, recover exactly what it was trained on.
        recovered = manager.load(record.dataset_version)
        assert recovered.version_id == version.version_id
        assert len(recovered.samples) == len(version.samples)
        assert "grouped by group" in record.model_card()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
