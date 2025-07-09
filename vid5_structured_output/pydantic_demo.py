from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    
    name : str
    age : Optional[int] = None
    email: EmailStr
    cgpa : float = Field(gt= 0, lt = 10, default = 5, description="a decimal value representing cgpa")
    
    
new_student = {'name' : 'nitish', 'age' : '32', 'email' : 'bd@gmail.com'}
# pydantic is smart enough to convert string '32' as int

student = Student(**new_student) 
# ** is used to unpack the dictionary from new_student

# Field function is used to put range in which our value should lie
# EmailStr helps to validate emails

print(student)