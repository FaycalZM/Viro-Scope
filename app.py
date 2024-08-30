from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from fluvio import Fluvio
from src.data_ingestion import fetch_google_trends_data, save_raw_data
import os

'''
This is a custom connector that fetches 
google trends data and integrates it
into our Fluvio data pipeline
'''


TOPIC_NAME = "google-trends-data"
BUFFER_MAX_SIZE = 16304


app = Flask(__name__)

# fluvio initialization
fluvio = Fluvio.connect()
# use CLI to create the topic (fluvio python sdk does not support topics creation yet)
# os.popen('fluvio topic create google-trends-data')
producer = fluvio.topic_producer(TOPIC_NAME)


# fetch and push google trends data to the topic
def fetch_and_push_google_trends_data():
    keywords = ["flu"]
    data = fetch_google_trends_data(keywords)
    # save_raw_data(data)

    if not data.empty:
        # convert the trends data to json string
        data_json = data.reset_index().to_json(orient='records', date_format='iso')

        # the buffer max_size is 16384 bytes, so it's not possible to send the data directly -> divide and conquer
        # convert string data to bytes
        byte_data = data_json.encode('utf-8')

        # split the bytes data into chunks of chunk_size bytes
        chunk_size = BUFFER_MAX_SIZE
        byte_chunks = [byte_data[i:i + chunk_size]
                       for i in range(0, len(byte_data), chunk_size)]

        # convert each chunk back to string
        string_chunks = [chunk.decode('utf-8', errors='ignore')
                         for chunk in byte_chunks]

        for chunk in string_chunks:
            # push data to the topic
            producer.send_string(chunk)
        # flush the last entry
        producer.flush()
        print(f"Data pushed to {TOPIC_NAME} topic")


# now we schedule the fetch_and_push_google_trends_data function to run every 24 hours
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_and_push_google_trends_data,
                  trigger="interval", hours=24)
scheduler.start()


# route for manual data ingestion
@app.route("/fetch-trends", methods=["GET"])
def fetch_google_trends():
    fetch_and_push_google_trends_data()
    return "Trends fetched and pushed to Fluvio successfully"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


# shut the scheduler down when on app exit
@app.teardown_appcontext
def shutdown_scheduler(exception=None):
    if scheduler.running:
        scheduler.shutdown()
