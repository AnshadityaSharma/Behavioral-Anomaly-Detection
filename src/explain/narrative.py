"""Turns the top contributing features for an alert into one readable sentence.

Deterministic templating rather than a generation call: this runs for every
alert in the queue, and an analyst reading the same alert twice should get the
same wording.
"""

TYPE_PHRASING = {
    "brute_force": "brute force",
    "credential_stuffing": "credential stuffing",
    "impossible_travel": "impossible travel",
    "lateral_movement": "lateral movement",
    "device_spoofing": "device spoofing",
    "insider_drift": "insider drift",
    "low_and_slow_exfil": "low-and-slow exfiltration",
}


def _fmt_int(value):
    return f"{int(round(value)):,}"


# SHAP can push a feature toward the predicted class even when that feature's
# own value is unremarkable (trees split on low values too). Showing an analyst
# "a 0% failure rate" as grounds for an alert is worse than showing nothing, so
# each phrase only renders above a value that is actually worth reading.
def _phrase_geo_velocity(value, row):
    if value < 100:
        return None
    return f"an implied travel speed of {_fmt_int(value)} km/h since the previous session"


def _phrase_geo_distance_home(value, row):
    if value < 100:
        return None
    city = row.get("geo_city")
    where = f" ({city})" if isinstance(city, str) and city and city != "unknown-transit" else ""
    return f"a login {_fmt_int(value)} km from this entity's usual location{where}"


def _phrase_new_resource(value, row):
    if not value:
        return None
    return f"first-ever access to {row.get('resource_accessed', 'this resource')} by this entity"


def _phrase_device_mismatch(value, row):
    if not value:
        return None
    os_name = row.get("device_os")
    detail = f" (now reporting {os_name})" if isinstance(os_name, str) and os_name else ""
    return f"a device fingerprint that does not match this entity's prior sessions{detail}"


def _phrase_ip_distinct_entities(value, row):
    if value < 2:
        return None
    return f"a source IP being used against {_fmt_int(value)} other accounts in the same 10-minute window"


def _phrase_ip_failure_rate(value, row):
    if value < 0.2:
        return None
    return f"a {value:.0%} authentication failure rate from this source IP"


def _phrase_ip_session_count(value, row):
    if value < 5:
        return None
    return f"{_fmt_int(value)} sessions from this source IP in 10 minutes"


def _phrase_entity_failed(value, row):
    if value < 2:
        return None
    return f"{_fmt_int(value)} failed logins for this entity within 10 minutes"


def _phrase_resource_breadth(value, row):
    if value < 8:
        return None
    return f"{_fmt_int(value)} different resources touched in 24 hours, well above this entity's norm"


def _phrase_resource_breadth_7d(value, row):
    if value < 6:
        return None
    return f"{_fmt_int(value)} different resources gradually touched over the trailing week"


def _phrase_hour_z(value, row):
    if abs(value) < 1.5:
        return None
    direction = "later" if value > 0 else "earlier"
    return f"a login {abs(value):.1f} standard deviations {direction} than this entity's usual hours"


def _phrase_duration_z(value, row):
    if abs(value) < 1.5:
        return None
    direction = "longer" if value > 0 else "shorter"
    return f"a session {abs(value):.1f} standard deviations {direction} than this entity's norm"


def _phrase_privileged_command(value, row):
    if not value:
        return None
    return "privileged commands executed during the session"


def _phrase_command_length(value, row):
    if value < 5:
        return None
    return f"an unusually long command sequence ({_fmt_int(value)} actions)"


def _phrase_auth_failed(value, row):
    if not value:
        return None
    return "a failed authentication on this session"


def _phrase_idle_gap(value, row):
    if value >= 9999:
        return "no prior activity on record for this entity"
    if value < 72:
        return None
    return f"a {_fmt_int(value)}-hour gap since this entity was last active"


PHRASE_BUILDERS = {
    "geo_velocity_kmh": _phrase_geo_velocity,
    "geo_distance_from_home_km": _phrase_geo_distance_home,
    "is_new_resource": _phrase_new_resource,
    "is_device_mismatch": _phrase_device_mismatch,
    "ip_distinct_entities_10min": _phrase_ip_distinct_entities,
    "ip_failure_rate_10min": _phrase_ip_failure_rate,
    "ip_session_count_10min": _phrase_ip_session_count,
    "entity_failed_count_10min": _phrase_entity_failed,
    "entity_resource_breadth_24h": _phrase_resource_breadth,
    "entity_resource_breadth_7d": _phrase_resource_breadth_7d,
    "hour_of_day_z": _phrase_hour_z,
    "session_duration_z": _phrase_duration_z,
    "has_privileged_command": _phrase_privileged_command,
    "command_sequence_length": _phrase_command_length,
    "auth_failed": _phrase_auth_failed,
    "hours_since_last_session": _phrase_idle_gap,
}


def build_phrases(contributions, row):
    phrases = []
    for feature, value in contributions:
        builder = PHRASE_BUILDERS.get(feature)
        if builder is None:
            continue
        phrase = builder(value, row)
        if phrase:
            phrases.append(phrase)
    return phrases


def _join_phrases(phrases):
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]}, combined with {phrases[1]}"
    return f"{phrases[0]}, combined with {phrases[1]} and {phrases[2]}"


def compose_sentence(predicted_type, contributions, row, cold_start_note_text=None):
    phrases = build_phrases(contributions, row)

    if not phrases and predicted_type in TYPE_PHRASING:
        base = (
            f"Flagged as {TYPE_PHRASING[predicted_type]} on a combination of weak signals, "
            "with no single feature standing out - see the SHAP breakdown"
        )
    elif not phrases:
        base = "Flagged by the ensemble, but no single feature dominated the score"
    elif predicted_type in TYPE_PHRASING:
        base = f"Flagged as {TYPE_PHRASING[predicted_type]} due to {_join_phrases(phrases)}"
    else:
        base = f"Elevated risk with no clear attack pattern, driven by {_join_phrases(phrases)}"

    sentence = base + "."
    if cold_start_note_text:
        sentence += f" {cold_start_note_text}"
    return sentence
