import lgpio
import time

class ServoMotor:
    def __init__(self, gpio, chip=0, min_us=500, max_us=2500):
        self.chip = lgpio.gpiochip_open(chip)
        self.gpio = gpio
        self.frequency = 50
        self.period_ns = int(1e9 / self.frequency)
        self.min_us = min_us
        self.max_us = max_us
        lgpio.gpio_claim_output(self.chip, self.gpio)

    def move_to(self, angle):
        if angle < 0: angle = 0
        if angle > 180: angle = 180

        pulse_us = self.min_us + (angle / 180.0) * (self.max_us - self.min_us)

        duty_cycle = (pulse_us * 1000) / self.period_ns * 100
        lgpio.tx_pwm(self.chip, self.gpio, self.frequency, duty_cycle)
        time.sleep(0.5)

    def open(self):
        self.move_to(180)

    def close(self):
        self.move_to(0)
        self.cleanup()

    def cleanup(self):
        lgpio.gpio_free(self.chip, self.gpio)
        lgpio.gpiochip_close(self.chip)

# Testing
# if __name__ == "__main__":
#     servo = ServoMotor(gpio=13, min_us=500, max_us=2500)
#     print("Opening fully...")
#     servo.open()
#     time.sleep(2)

#     print("Closing fully...")
#     servo.close()
#     time.sleep(2)
