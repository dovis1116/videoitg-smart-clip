from videoitg_smart_clip.badcase.taxonomy import BADCASE_TYPES, normalize_badcase_type


def test_badcase_taxonomy_rejects_unclassified_labels():
    assert "retrieval_miss" in BADCASE_TYPES
    assert normalize_badcase_type("wrong_event") == "wrong_event"
    assert normalize_badcase_type("grounding_problem") == "unclassified"
