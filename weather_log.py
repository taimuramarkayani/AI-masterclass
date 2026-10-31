import os
import sys
import json
import requests
 
# ---------- Settings ----------
CITY = "Lahore"   # change to your city (must be in CITIES below)
CITIES = {
    "Lahore": (31.52, 74.32),
    "Karachi": (24.86, 67.01),
    "Islamabad": (33.69, 73.05),
}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
 
 
# ---------- 1. Get weather data (your code, not the AI) ----------
def get_weather(city):
    lat, lon = CITIES[city]
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max"
        "&timezone=auto&past_days=5&forecast_days=1"
    )
    d = requests.get(url, timeout=30).json()["daily"]
    days = []
    for i in range(5):  # the first 5 days; today is not finished, so we skip it
        days.append({
            "date": d["time"][i],
            "high": d["temperature_2m_max"][i],
            "low": d["temperature_2m_min"][i],
            "wind": d["wind_speed_10m_max"][i],
        })
    return {
        "city": city,
        "days": days,
        "avg_high": round(sum(x["high"] for x in days) / 5, 1),
        "avg_low": round(sum(x["low"] for x in days) / 5, 1),
        "peak_wind": max(x["wind"] for x in days),
    }
 
 
# ---------- 2. Talk to the AI ----------
def ask_groq(messages, **extra):
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
        json={"model": MODEL, "messages": messages, **extra},
        timeout=60,
    )
    if r.status_code != 200:
        raise SystemExit("Groq error: " + r.text)
    return r.json()["choices"][0]["message"]
 
 
def get_json(prompt, keys):
    # Ask for JSON. If the reply is broken, try once more.
    for attempt in range(2):
        text = ask_groq(
            [{"role": "system", "content": "Reply with JSON only. No extra text."},
             {"role": "user", "content": prompt}],
            temperature=0.2,
        )["content"].strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("Bad JSON, trying again")
            continue
        if all(k in data for k in keys):
            return data
        print("Missing fields, trying again")
    raise SystemExit("The AI did not return valid JSON")
 
 
# ---------- 3. The daily job ----------
def run_job(dry=False):
    data = get_weather(CITY)
    table = "\n".join(
        f"{x['date']}: high {x['high']}C, low {x['low']}C, max wind {x['wind']} km/h"
        for x in data["days"]
    )
    prompt = (
        f"Weather for {CITY}, last 5 days:\n{table}\n\n"
        "Return JSON with exactly these keys: "
        "trend (warming, cooling or steady), "
        "wind_risk (low, medium or high), "
        "advice (one sentence), "
        "send_alert (true only if something unusual needs attention, otherwise false), "
        "reason (one sentence). "
        "Use only the numbers above. Do not invent anything."
    )
    result = get_json(prompt, ["trend", "wind_risk", "advice", "send_alert", "reason"])
    print(result)
 
    payload = {
        "city": CITY,
        "period": data["days"][0]["date"] + " to " + data["days"][-1]["date"],
        "avg_max": data["avg_high"],
        "avg_min": data["avg_low"],
        "peak_wind": data["peak_wind"],
        "trend": result["trend"],
        "wind_risk": result["wind_risk"],
        "advice": result["advice"],
        "send_alert": result["send_alert"] is True,
        "reason": result["reason"],
    }
 
    if dry:
        print("Dry run. Nothing was sent to the sheet.")
        return
 
    response = requests.post(os.environ["SHEET_URL"], json=payload, timeout=30)
    print(response.status_code, response.text)
    if response.text != "OK":
        raise SystemExit("The sheet script said: " + response.text)
 
 
# ---------- 4. The agent: the AI chooses a tool, your code runs it ----------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the 5 day weather summary (average high, average low, peak wind) for one city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "enum": list(CITIES)}},
            "required": ["city"],
        },
    },
}]
 
 
def run_agent(question, max_steps=5):
    messages = [
        {"role": "system", "content": "You answer weather questions. Use the tool for facts. Never guess numbers."},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        msg = ask_groq(messages, tools=TOOLS, temperature=0.2)
        calls = msg.get("tool_calls")
        if not calls:
            return msg["content"]          # no tool needed, this is the final answer
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            print("STEP", step, "->", name, args)       # log every decision
            if name == "get_weather" and args.get("city") in CITIES:
                data = get_weather(args["city"])
                result = {k: v for k, v in data.items() if k != "days"}
            else:
                result = {"error": "tool or city not allowed"}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return "Stopped: too many steps"
 
 
# ---------- Start here ----------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "job"
    if mode == "ask":
        print(run_agent(" ".join(sys.argv[2:])))
    elif mode == "dry":
        run_job(dry=True)
    else:
        run_job()