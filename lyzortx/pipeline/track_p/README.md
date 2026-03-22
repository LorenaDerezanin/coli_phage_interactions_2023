# Track P Presentation Artifacts

`python lyzortx/pipeline/track_p/run_track_p.py`

This command builds the TP01 digital phagogram:

1. `digital-phagogram`: generate a standalone interactive HTML dashboard plus CSV/JSON summaries under
   `lyzortx/generated_outputs/track_p/digital_phagogram/`

The dashboard uses the locked TG05 feature configuration and shows panel-model and deployment-realistic phage
rankings side by side for any strain in the panel. Each ranked phage row includes calibrated `P(lysis)`, a confidence
band, and the top SHAP feature drivers.
