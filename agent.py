class ModelBasedReflexAgent: 
    def __init__(self):
        self.current_temp = None
        self.current_room = None
        self.environment_data = {}
        self.memory_file = "memory.txt"

    def load_environment(self, filename="rooms.txt"):
        
        with open(filename, "r") as file:
            for line in file:
                room_name, temp_str, ac_status = line.strip().split(",")
                self.environment_data[room_name.strip()] = {
                    "temp": int(temp_str.strip()),
                    "ac": ac_status.strip()
                }
    
    def sensor(self, temperature, room): 
       
        self.current_temp = temperature
        self.current_room = room

    def performance(self):
        

        
        try:
            with open(self.memory_file, "r") as f:
                for line in f:
                    r, t, desired, action = line.strip().split(",")
                    if r == self.current_room and int(t) == self.current_temp:
                        return f"From Memory: {action}"
        except FileNotFoundError:
            pass

        
        if self.current_room not in self.environment_data:
            return f"Room '{self.current_room}' not found in environment."

        env_temp = self.environment_data[self.current_room]["temp"]
        ac_status = self.environment_data[self.current_room]["ac"]

       
        action = f"AC was {ac_status} at {env_temp}C"

      
        with open(self.memory_file, "a") as f:
            f.write(f"{self.current_room},{self.current_temp},24,{action}\n")

        return f"New Decision: {action}"

    def actuator(self):
        decision = self.performance()
        print(decision)




agent = ModelBasedReflexAgent()       
agent.load_environment("rooms.txt")      


agent.sensor(28, "kitchen")
agent.actuator()

agent.sensor(24, "bedroom")
agent.actuator()

agent.sensor(28, "kitchen")
agent.actuator()

agent.sensor(20, "study room")
agent.actuator()

agent.sensor(30, "garage")
agent.actuator()

agent.sensor(30, "garage")
agent.actuator()
