import os
import json
import requests
import gradio as gr

# Configure the API endpoint (override with env var PREDICT_API_URL)
API_URL = os.getenv("PREDICT_API_URL", "http://10.43.100.102:8013/predict")


def predict(
    elevation,
    aspect,
    slope,
    h_dist_hyd,
    v_dist_hyd,
    h_dist_road,
    hill_9am,
    hill_noon,
    hill_3pm,
    h_dist_fire,
    wilderness_area,
    soil_type,
):
    payload = {
        "Elevation": int(elevation),
        "Aspect": int(aspect),
        "Slope": int(slope),
        "Horizontal_Distance_To_Hydrology": int(h_dist_hyd),
        "Vertical_Distance_To_Hydrology": int(v_dist_hyd),
        "Horizontal_Distance_To_Roadways": int(h_dist_road),
        "Hillshade_9am": int(hill_9am),
        "Hillshade_Noon": int(hill_noon),
        "Hillshade_3pm": int(hill_3pm),
        "Horizontal_Distance_To_Fire_Points": int(h_dist_fire),
        "Wilderness_Area": wilderness_area,
        "Soil_Type": soil_type,
    }

    status = f"POST {API_URL}"
    try:
        resp = requests.post(API_URL, json=payload, timeout=20)
        status += f"\nStatus: {resp.status_code}"
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            # Fallback if not JSON
            data = {"raw": resp.text}
        # Try to surface a common field name if present
        pretty_summary = None
        for key in ("prediction", "pred", "class", "label", "result"):
            if isinstance(data, dict) and key in data:
                pretty_summary = f"**Model output ({key})**: {data[key]}"
                break
        if pretty_summary is None:
            pretty_summary = "**Model output**: (see JSON)"
        return payload, data, pretty_summary, status
    except requests.exceptions.RequestException as e:
        err = str(e)
        return payload, {"error": err}, "**Request failed** — check API_URL and server logs.", status


with gr.Blocks(title="Cover Type Predictor") as demo:
    gr.Markdown(
        """
        # 🌲 Cover Type Predictor — via `/predict`
        Set the inputs and click **Predict**. The app will POST the JSON body to your API and show the response.
        
        > Configure the endpoint with env var: `PREDICT_API_URL` (default: `http://localhost:8000/predict`).
        """
    )

    with gr.Row():
        with gr.Column():
            elevation = gr.Number(label="Elevation", value=2358, precision=0)
            aspect = gr.Number(label="Aspect", value=8, precision=0)
            slope = gr.Number(label="Slope", value=5, precision=0)
            h_dist_hyd = gr.Number(label="Horizontal_Distance_To_Hydrology", value=170, precision=0)
            v_dist_hyd = gr.Number(label="Vertical_Distance_To_Hydrology", value=19, precision=0)
            h_dist_road = gr.Number(label="Horizontal_Distance_To_Roadways", value=1354, precision=0)
            hill_9am = gr.Number(label="Hillshade_9am", value=214, precision=0)
            hill_noon = gr.Number(label="Hillshade_Noon", value=230, precision=0)
            hill_3pm = gr.Number(label="Hillshade_3pm", value=153, precision=0)
            h_dist_fire = gr.Number(label="Horizontal_Distance_To_Fire_Points", value=342, precision=0)

            wilderness_area = gr.Dropdown(
                label="Wilderness_Area",
                choices=[
                    "Rawah",
                    "Neota",
                    "Comanche",
                    "Cache",
                ],
                value="Cache",
            )

            soil_type = gr.Textbox(label="Soil_Type (e.g., C2717)", value="C2717")

            predict_btn = gr.Button("🚀 Predict", variant="primary")

        with gr.Column():
            gr.Markdown("### Request preview")
            req_json = gr.JSON()
            gr.Markdown("### Response from API")
            resp_json = gr.JSON()
            summary_md = gr.Markdown()
            status_md = gr.Markdown()

    predict_btn.click(
        fn=predict,
        inputs=[
            elevation,
            aspect,
            slope,
            h_dist_hyd,
            v_dist_hyd,
            h_dist_road,
            hill_9am,
            hill_noon,
            hill_3pm,
            h_dist_fire,
            wilderness_area,
            soil_type,
        ],
        outputs=[req_json, resp_json, summary_md, status_md],
        api_name="predict_cover_type",
    )

    gr.Examples(
        label="Examples",
        examples=[
            [2358, 8, 5, 170, 19, 1354, 214, 230, 153, 342, "Cache", "C2717"],
            [2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, "Rawah", "C7744"],
            [2920, 223, 11, 90, 13, 1973, 205, 251, 181, 1226, "Comanche", "C7756"],
            [3286, 60, 13, 0, 0, 2214, 230, 212, 113, 2846, "Rawah", "C8772"],

        ],
        inputs=[
            elevation,
            aspect,
            slope,
            h_dist_hyd,
            v_dist_hyd,
            h_dist_road,
            hill_9am,
            hill_noon,
            hill_3pm,
            h_dist_fire,
            wilderness_area,
            soil_type,
        ],
    )

    with gr.Accordion("cURL example", open=False):
        curl_md = gr.Markdown(
            f"""
```bash
curl -X POST {API_URL} \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "Elevation": 2358,
    "Aspect": 8,
    "Slope": 5,
    "Horizontal_Distance_To_Hydrology": 170,
    "Vertical_Distance_To_Hydrology": 19,
    "Horizontal_Distance_To_Roadways": 1354,
    "Hillshade_9am": 214,
    "Hillshade_Noon": 230,
    "Hillshade_3pm": 153,
    "Horizontal_Distance_To_Fire_Points": 342,
    "Wilderness_Area": "Cache",
    "Soil_Type": "C2717"
  }}'
```
            """
        )


demo.launch()
