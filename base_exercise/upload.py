import paho.mqtt.client as mqtt
import time


USERID = "admin"
PASSWORD = "password"

resp_callback = None

def on_connect(client, userdata, flags, rc):
    print("connected. Result coed is %d." % rc)
    client.subscribe(USERID+"/IMAGE/predict")


def on_message(client, userdata, msg):
    print("Received message form server.", msg.payload)

    # tmp = msg.payload.decode('utf-8')
    # str = tmp.split(":")

    # if resp_callback is not None:
    #     resp_callback(str[0], float(str[1]), int(str[2]))


def setup():
    global client

    client = mqtt.Client(transport='tcp')
    client.username_pw_set(USERID, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("127.0.0.1", 61613, 30)
    # client.connect("172.25.98.192", 61680, 30)
    # client.loop_start()
    client.loop_start()

# def load_image(file_name):
#     with open(file_name, "rb") as f:
#         data = f.read()
#     return data


def send_image(file_name):
    # img = load_image(file_name)
    client.publish(USERID+"/IMAGE/classify", "1111")


def resp_handler(label, prob, index):
    print("\n -- Response -- \n\n")
    print("Label: %s" % label)
    print("Probability: %3.4f" % prob)
    print("Index: %d" % index)


def main():
    global resp_callback

    setup()
    # resp_callback = resp_handler
    # print("Waiting 2 seconds before sending.")
    # time.sleep(2)
    # print("Sending tulip.jpg")
    # send_image('tulip.jpg')
    # print("Done. Waiting 5 seconds before sending")
    # time.sleep(5)
    # print("Sending tulip2.jpg")
    # send_image("tulip2.jpg")

    while True:
        pass


if __name__ == '__main__':
    main()
