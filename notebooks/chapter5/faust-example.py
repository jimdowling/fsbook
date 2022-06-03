import faust
import hopsworks
import ssl

import hsfs

HOST="34.78.148.2"
PROJECT="demo_fs_meb10000"

connection = hsfs.connection(
    host=HOST,
    project=PROJECT,
    api_key_value="9tybM5t8Vj5ZPH6L.badVzfTTdelhCTZLDHthlHL7ITfDTI9IRsplEQ91ViHmBdn5JFDXJsJ47s7UtklL",
    engine="python"
)
fs = connection.get_feature_store()

BASE_PATH = "/tmp/" + HOST + "/" + PROJECT + "/"
CA_FILE = BASE_PATH + "ca_chain.pem"
CERT_FILE = BASE_PATH + "client_cert.pem"
KEY_FILE = BASE_PATH + "client_key.pem"
PASSWORD_FILE = BASE_PATH + "material_passwd"
text_file = open(PASSWORD_FILE, "r")
PASSWORD=text_file.read()
text_file.close()


ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=CA_FILE)
ssl_context.load_cert_chain(CERT_FILE, keyfile=KEY_FILE, password=PASSWORD)
# ssl_context = kafka.get_ssl_context()

# raw -> avro??
BROKER_URL = "kafka://" + HOST + ":9092"
app = faust.App('empty', broker=BROKER_URL, store=' memory://', value_serializer='json', broker_credentials=ssl_context)

cc_topic = app.topic('empty', key_type=str, value_type=str)

# This sends events every 1 second to the process method above.
@app.timer(interval=10.0)
async def topic_writer(app):
    await cc_topic.send(
        value="hello",
    )

if __name__ == '__main__':
    app.main()
