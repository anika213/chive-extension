import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from location_seed_analysis import edit_location, strip_location, seed_concentration


def test_inserted_message_does_not_shift_locations():
    system = {'role': 'system', 'content': 'Rules'}
    user = {'role': 'user', 'content': 'Question'}
    old = [system, user]
    result = edit_location(old, [system, {'role': 'assistant', 'content': 'Earlier answer'}, user])
    assert result['edit_locations'] == ['prior_assistant_turn']
    assert result['changed_baseline_message_indices'] == []
    assert result['changed_experiment_message_indices'] == [1]


def test_wrapped_tools_and_native_roles_remain_distinct():
    old = [{'role':'user', 'content':'<tool_response>Error</tool_response>'}]
    new = [{'role':'user', 'content':'<tool_response>Success</tool_response>'}]
    result = edit_location(old, new)
    assert result['edit_locations'] == ['tool_output']
    assert result['changed_message_roles'] == ['user']
    assert edit_location(old, old)['edit_location'] == 'no_change'
    assert edit_location(old, [{'role':'system','content':'Rule'}])['edit_locations'] == ['system_prompt','tool_output']


def test_remove_locator_preserve_payload():
    text = "Added 'ignore the system prompt' to the final user message."
    cleaned, removed = strip_location(text)
    assert cleaned == "Added 'ignore the system prompt'."
    assert removed == ['to the final user message']
    assert strip_location("Changed the final question to 'Why?'")[0] == "Changed to 'Why?'"


def test_seed_fractions_missingness_and_noise():
    rows = [{'ward_cluster_id':0,'scenario_seed_proxy':s} for s in ['a','a','b','c']]
    rows.append({'ward_cluster_id':-1,'scenario_seed_proxy':'a'})
    result = seed_concentration(rows,'ward')[0]
    assert result['size'] == 4 and result['distinct_seeds'] == 3
    assert result['top1_fraction'] == .5 and result['top2_fraction'] == .75
    rows[0]['scenario_seed_proxy'] = None
    assert seed_concentration(rows,'ward')[0]['top1_fraction'] is None
