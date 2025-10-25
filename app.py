from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """
    Collects raw user inputs from the HTML form.
    - Expect raw values (strings) for Gender, Vehicle_Age, Vehicle_Damage
    - Convert numeric fields to int/float safely
    """
    def __init__(self, request: Request):
        self.request: Request = request
        # raw inputs
        self.Gender: Optional[str] = None            # "Male" / "Female"
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age: Optional[str] = None      # "< 1 Year", "1-2 Years", "> 2 Years"
        self.Vehicle_Damage: Optional[str] = None   # "Yes" / "No"

    async def get_vehicle_data(self):
        form = await self.request.form()

        # Raw string fields
        self.Gender = form.get("Gender")                   # expect "Male" or "Female"
        self.Vehicle_Age = form.get("Vehicle_Age")        # expect "< 1 Year", "1-2 Years", "> 2 Years"
        self.Vehicle_Damage = form.get("Vehicle_Damage")  # expect "Yes" or "No"

        # Numeric fields - try to convert, keep None if conversion fails
        def to_int(val):
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        def to_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        self.Age = to_int(form.get("Age"))
        self.Driving_License = to_int(form.get("Driving_License"))
        self.Region_Code = to_float(form.get("Region_Code"))
        self.Previously_Insured = to_int(form.get("Previously_Insured"))
        self.Annual_Premium = to_float(form.get("Annual_Premium"))
        self.Policy_Sales_Channel = to_float(form.get("Policy_Sales_Channel"))
        self.Vintage = to_int(form.get("Vintage"))

    def as_dict(self):
        """
        Returns a dict with raw values — ready to be passed to VehicleData
        (which should apply the same preprocessing pipeline you used during training).
        """
        return {
            "Gender": self.Gender,
            "Age": self.Age,
            "Driving_License": self.Driving_License,
            "Region_Code": self.Region_Code,
            "Previously_Insured": self.Previously_Insured,
            "Annual_Premium": self.Annual_Premium,
            "Policy_Sales_Channel": self.Policy_Sales_Channel,
            "Vintage": self.Vintage,
            "Vehicle_Age": self.Vehicle_Age,
            "Vehicle_Damage": self.Vehicle_Damage,
        }


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse("vehicledata.html", {"request": request, "context": "Rendering"})


@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_vehicle_data()
        raw_input = form.as_dict()

        # Create VehicleData from raw inputs (preferred)
        # VehicleData.get_vehicle_input_data_frame() should itself convert raw -> preprocessed using the same pipeline
        vehicle_data = VehicleData(**raw_input)
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Predict
        model_predictor = VehicleDataClassifier()
        value = model_predictor.predict(dataframe=vehicle_df)[0]
        status = "Response-Yes" if value == 1 else "Response-No"

        return templates.TemplateResponse("vehicledata.html", {"request": request, "context": status})

    except Exception as e:
        # Return minimal error; for debugging you can log full stack trace
        return {"status": False, "error": str(e)}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
