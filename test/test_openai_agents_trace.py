# test/test_openai_agents_trace.py
import pytest
import os
import asyncio
from typing import List, Optional, Literal
import time
import uuid

# Import required libraries with error handling
try:
    from pydantic import BaseModel, Field
    from agents import Agent, ModelSettings, Runner
    from ragmetrics.api import ragmetrics_client
except ImportError as e:
    pytest.fail(f"Required dependencies not found: {e}")

# Define Pydantic models
class PriceBreakdown(BaseModel):
    federal_excise_tax_amount: Optional[float] = Field(
        None,
        description="Federal excise tax (FET) amount, do not calculate, provide null if not found",
    )
    segment_fee_amount: Optional[float] = Field(
        None,
        description="The segment fee, segment tax amount or federal segment tax (may have alternative names like Taxable Passenger Segment), do not calculate, provide null if not found",
    )

class FlightLeg(BaseModel):
    leg_number: int = Field(
        ..., description="Chronological order number of the flight leg"
    )
    departure_date: str = Field(
        ...,
        description="Departure date, do NOT use any date from the email headers (Date: ...), only what is listed in the Quote PDF. YYYY-MM-DD format",
    )
    departure_time: Optional[str] = Field(
        None,
        description="Departure time in HH:mm format. If unavailable, TBD or TBA, use null. NEVER write TBD or TBA",
    )
    origin_icao: str = Field(
        ...,
        description="Departure airport location ICAO of the flight leg, such as KLAX, KJFK, KCMH, KSLC, etc. NEVER write a city name here",
    )
    destination_icao: str = Field(
        ...,
        description="Arrival airport location of the flight leg ICAO of the flight leg, such as KLAX, KJFK, KCMH, KSLC, etc",
    )
    estimated_time_en_route: str = Field(
        ...,
        description="Duration of the flight leg, HH:mm format, empty string if not provided. It might be specified under, for example, ETE, Est Hours, duration or something similar. Do not calculate the duration if not provided",
    )
    estimated_time_en_route_hours: int = Field(
        ...,
        description="Duration of the flight leg in hour(s), use zero (0) if not provided. It might be specified under, for example, ETE, Est Hours, duration or something similar. Do not calculate the duration if not provided",
    )
    estimated_time_en_route_minutes: int = Field(
        ...,
        description="Duration of the flight leg in minute(s), use zero (0) if not provided. It might be specified under, for example, ETE, Est Hours, duration or something similar. Do not calculate the duration if not provided",
    )
    arrival_date: str = Field(
        ...,
        description="Arrival date, do NOT use any date from the email headers (Date: ...), only what is listed in the Quote PDF, YYYY-MM-DD format",
    )
    arrival_time: Optional[str] = Field(
        None,
        description="Arrival time in HH:mm format. If unavailable, TBD or TBA, use null. NEVER write TBD or TBA",
    )
    passenger_count: int = Field(
        ...,
        description="Number of passengers/seats/pax on the flight leg, 0 if not available or not specified",
    )
    aircraft_make: Optional[
        Literal[
            "Bombardier",
            "IAI",
            "Embraer",
            "Airbus",
            "Gulfstream",
            "Dassault",
            "Mitsubishi Diamond",
            "Lockheed",
            "Sabreliner",
            "Cessna",
            "Honda",
            "Pilatus",
            "Saab",
            "McDonnell Douglas",
            "Beech",
            "Piper",
            "Daher",
            "Nextant",
            "Boeing",
            "Eclipse",
            "Cirrus",
            "Hawker",
            "Helicopter",
        ]
    ] = Field(
        None,
        description="Make of the aircraft as defined in the enum. Set as null if not available. NEVER use a Category Name here. (Light Jet, etc.). If aircraft_make_and_model is filled out, this should be filled out",
    )
    aircraft_model: Optional[str] = Field(
        None,
        description="Model of the aircraft. Set as null if not available. NEVER use a Category Name here such as Light Jet. If aircraft_make_and_model is filled out, this should be filled out",
    )
    aircraft_make_and_model: Optional[str] = Field(
        None,
        description="Make and Model of the aircraft, e.g. Gulfstream G-IV (be sure to include the dash and roman numeral if applicable). Set as null if not available. NEVER use a Category Name here such as Light Jet",
    )
    aircraft_category_name: Optional[str] = Field(
        None,
        description="Category of the aircraft type or size (Light Jet, etc.), if specified. Set as null if not available. There should only be one category name.",
    )
    nautical_miles: Optional[float] = Field(
        None,
        description="Distance in nautical miles between the origin and destination airports for the flight leg",
    )
    tail_number: Optional[str] = Field(
        None,
        description="The tail number of the airplane. Tail numbers may not exceed five characters in addition to the prefix letter N. These characters may be: One to five numbers (N12345), One to four numbers followed by one letter (N1234Z), One to three numbers followed by two letters (N123AZ). Do not make up tail numbers if not provided, use null.",
    )

class QuoteData(BaseModel):
    has_availability: bool = Field(
        ...,
        description="Determines whether the operator has availability for the requested flight or an alternative flight. If the email contains a quote with pricing details (either in the email body or as an attachment), assume availability is confirmed unless there is an explicit statement that no availability exists. General disclaimers such as 'subject to availability' or 'does not guarantee availability' do not override the assumption of availability unless a direct statement of unavailability is present.",
    )
    is_email_chatter: bool = Field(
        ...,
        description="Determines whether the email contains chatter, general communication, and is not a quote PDF or email quote or a decline of availability. A chatter email means the email is a question, information request, thank you note, etc. This should be set to false if the email contains a quote with pricing details (either in the email body or as an attachment) or specifically declines availability.",
    )
    operator_name: str = Field(
        ...,
        description="Name of the flight operator company (it will NEVER be Goodwin)",
    )
    operator_email_domain: str = Field(
        ..., description="Email domain of the operator company"
    )
    emails_listed: List[str] = Field(
        ..., description="Array of email addresses of the recipients of the quote"
    )
    operator_external_quote_id: str = Field(
        ...,
        description="Unique external identifier for the operator's quote, if available, it will NEVER start with QR-",
    )
    is_price_provided: bool = Field(
        ..., description="Is the price provided in the quote or email?"
    )
    price_breakdown: PriceBreakdown = Field(
        ..., description="Breakdown of price components"
    )
    price_total: float = Field(
        ...,
        description="Total price (subtotal and other fees and taxes) for the quote, if the total price is provided, 0 if not available",
    )
    price_currency: str = Field(..., description="Currency of the prices in the quote")
    filename: str = Field(
        ...,
        description="Filename of attachment the quote information is retrieved from. Leave as null if quote data is written in the main email body.",
    )
    flight_legs: List[FlightLeg] = Field(
        ...,
        description="Array of objects containing details of each flight leg, AI must ensure the legs are in chronological order and ALL legs are present in the array. Do not source information from PDFs that are not a quote, such as (wyvern, spec sheet, FAA docs, etc.)",
    )

class QuoteDataItem(BaseModel):
    quote_data: QuoteData

class QuotesWithFlightLegs(BaseModel):
    quote_data_list: List[QuoteDataItem] = Field(
        ...,
        description="Array of objects containing the quote data based on the schema extracted from an email possibly containing a ballpark price or quote PDF attachment with price and flight leg details. A single email/PDF may contain one or more quotes.",
    )

def test_openai_agents_trace(ragmetrics_test_client=None):
    """Tests creating a trace using OpenAI Agents SDK, saving it, and then downloading it."""
    # Client setup
    client = ragmetrics_test_client
    
    # Check for RagMetrics API key
    if not client:
        api_key = os.environ.get("RAGMETRICS_API_KEY")
        if not api_key:
            raise ValueError("RAGMETRICS_API_KEY environment variable is required for this test")
        
        # Login with the API key
        ragmetrics_client.login(key=api_key)
    
    # Sample flight data for testing
    sample_email = """
    From: charter@skyairways.com
    To: client@example.com
    Subject: Your Flight Quote

    Flight Date: 2023-10-01
    Flight Time: 10:00
    Departure: KLAX
    Arrival: KJFK
    Duration: 05:00
    Aircraft: Gulfstream G-IV
    Price: $10,000 USD
    Operator: Sky Airways
    Quote ID: SKY-1234
    """
    
    # Create an OpenAI agent
    agent = Agent(
        name="Assistant",
        instructions="As an AI assistant, your role involves analyzing details from private charter flight itineraries and quotes, including information such as flight leg order, origin and destination locations, departure and arrival times, flight duration, aircraft details, and pricing. Your task is to transform this information into a structured JSON format, suitable for database storage, according to the provided schema.",
        model="gpt-4o",
        model_settings=ModelSettings(temperature=0.0),
        output_type=QuotesWithFlightLegs
    )
    
    # Monitor the Runner class to track all agent calls
    ragmetrics_client.monitor(Runner, metadata={
        "test_name": "test_openai_agents_trace"
    })
    
    # Run the agent
    result = Runner.run_sync(agent, input=sample_email)
    
    # Assert the result structure and values
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'final_output'), "Result should have final_output"
    assert hasattr(result.final_output, 'quote_data_list'), "Result should have quote_data_list"
    assert len(result.final_output.quote_data_list) == 1, "Should have exactly one quote"
    
    quote_data = result.final_output.quote_data_list[0].quote_data
    assert quote_data.operator_name == "Sky Airways"
    assert quote_data.operator_email_domain == "skyairways.com"
    assert quote_data.operator_external_quote_id == "SKY-1234"
    assert quote_data.price_total == 10000.0
    assert quote_data.price_currency == "USD"
    
    # Assert flight leg details
    assert len(quote_data.flight_legs) == 1, "Should have exactly one flight leg"
    flight_leg = quote_data.flight_legs[0]
    assert flight_leg.origin_icao == "KLAX"
    assert flight_leg.destination_icao == "KJFK"
    assert flight_leg.departure_date == "2023-10-01"
    assert flight_leg.departure_time == "10:00"
    assert flight_leg.aircraft_make == "Gulfstream"
    assert flight_leg.aircraft_model == "G-IV"

if __name__ == "__main__":
    test_openai_agents_trace() 