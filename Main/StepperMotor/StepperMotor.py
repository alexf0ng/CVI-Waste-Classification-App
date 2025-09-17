import time

class StepperMotor:
    def __init__(self, in1, in2, in3, in4, raspberryPi, chip=0, test_material=None):
        if raspberryPi:
            import lgpio
            self.steps = 0
            self.IN1, self.IN2, self.IN3, self.IN4 = in1, in2, in3, in4
            self.chip = lgpio.gpiochip_open(chip)
            for pin in [self.IN1, self.IN2, self.IN3, self.IN4]:
                lgpio.gpio_claim_output(self.chip, pin)
            self.sequence = [
                [1,0,0,0],[1,1,0,0],[0,1,0,0],[0,1,1,0],
                [0,0,1,0],[0,0,1,1],[0,0,0,1],[1,0,0,1]
            ]
            self.steps_per_degree = 512 / 360

            if test_material:
                self.move_material(test_material)
        else:
            pass

    def spin(self, steps, delay=0.001, direction=1):
        seq = self.sequence if direction>0 else list(reversed(self.sequence))
        for _ in range(steps):
            for step in seq:
                lgpio.gpio_write(self.chip, self.IN1, step[0])
                lgpio.gpio_write(self.chip, self.IN2, step[1])
                lgpio.gpio_write(self.chip, self.IN3, step[2])
                lgpio.gpio_write(self.chip, self.IN4, step[3])
                time.sleep(delay)

    def move_material(self, material, delay=0.001):
        angles = {"plastic":0, "steel":90, "paper":180, "other":270}
        steps = int(angles.get(material.lower(),0) * self.steps_per_degree)
        if steps>0:
            self.spin(steps, delay)
            self.steps = steps

    def back_origin(self, delay=0.001):
        if self.steps>0:
            self.spin(self.steps, delay, direction=-1)

        self.cleanup()

    

    def cleanup(self):
        for pin in [self.IN1,self.IN2,self.IN3,self.IN4]:
            lgpio.gpio_write(self.chip, pin,0)
        lgpio.gpiochip_close(self.chip)

