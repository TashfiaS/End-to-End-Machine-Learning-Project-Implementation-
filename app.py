import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from us_visa.constants import APP_HOST, APP_PORT
from us_visa.pipeline.prediction_pipeline import USvisaClassifier, USvisaData
from us_visa.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "usvisa.html", {"request": request, "context": "Rendering"}
    )


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful! Model has been pushed to S3.")
    except Exception as e:
        return Response(f"Error during training: {e}")


@app.post("/predict")
async def predict_route(request: Request):
    try:
        form = await request.form()

        usvisa_data = USvisaData(
            continent=form.get("continent"),
            education_of_employee=form.get("education_of_employee"),
            has_job_experience=form.get("has_job_experience"),
            requires_job_training=form.get("requires_job_training"),
            no_of_employees=int(form.get("no_of_employees")),
            yr_of_estab=int(form.get("yr_of_estab")),
            region_of_employment=form.get("region_of_employment"),
            prevailing_wage=float(form.get("prevailing_wage")),
            unit_of_wage=form.get("unit_of_wage"),
            full_time_position=form.get("full_time_position"),
        )

        df = usvisa_data.get_usvisa_input_data_frame()
        classifier = USvisaClassifier()
        prediction = classifier.predict(dataframe=df)

        context = f"Visa Status Prediction: {prediction}"
        return templates.TemplateResponse(
            "usvisa.html", {"request": request, "context": context}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": f"Prediction error: {e}"},
        )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", APP_PORT))
    uvicorn.run(app, host=APP_HOST, port=port)
