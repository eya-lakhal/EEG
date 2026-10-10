"""Inventaire des widgets et sorties de chaque page (pour comparer l'ancienne et la nouvelle interface)."""
import sys, json, warnings; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest

def sig_widgets(at):
    out = []
    def add(kind, items, fn):
        for w in items:
            try: out.append((kind,) + fn(w))
            except Exception as e: out.append((kind, "ERR", str(e)[:40]))
    add("slider", at.slider, lambda w: (w.min, w.max, w.step, str(w.value)))
    add("select_slider", at.select_slider, lambda w: (tuple(map(str, w.options)), str(w.value)))
    add("selectbox", at.selectbox, lambda w: (len(w.options), str(w.value)[:30]))
    add("radio", at.radio, lambda w: (tuple(map(str, w.options)), str(w.value)))
    add("checkbox", at.checkbox, lambda w: (str(w.value),))
    add("toggle", at.toggle, lambda w: (str(w.value),))
    add("number_input", at.number_input, lambda w: (w.min, w.max, w.step, w.value))
    add("multiselect", at.multiselect, lambda w: (len(w.options), tuple(map(str, w.value))))
    return sorted(out, key=str)

def sig_outputs(at):
    return {"metrics": sorted((m.label, str(m.value)) for m in at.metric),
            "dataframes": [tuple(d.value.shape) for d in at.dataframe],
            "tabs": len(at.tabs), "exceptions": [e.value[:120] for e in at.exception],
            "buttons": sorted(b.label for b in at.button)}

if __name__ == "__main__":
    script, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]       # mode: old | new
    at = AppTest.from_file(script, default_timeout=3000).run()
    res = {}
    for i in range(7):
        if mode == "old": at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[i]).run()
        else:
            at.session_state["step"] = i; at.run()
        res[i] = {"widgets": sig_widgets(at), "outputs": sig_outputs(at)}
        print("page", i, len(res[i]["widgets"]), "widgets", flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
