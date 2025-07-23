from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start_engine(self):
        print(f"{self.brand} {self.model}'s engine started.")

class Driver:
    def __init__(self, name):
        self.name = name

    def drive(self, car: Car):
        print(f"{self.name} is driving the {car.brand} {car.model}.")
        car.start_engine()

# Example usage
my_car = Car("Toyota", "Corolla")
driver = Driver("John")

driver.drive(my_car)

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])