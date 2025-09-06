from enum import Enum

class PenguinsType(Enum):
    ADELIE = ("Adelie",0)
    CHINSTRAP = ("Chinstrap",1)
    GENTOO = ("Gentoo",2)
    
    def get_penguins_by_value(value):
        for penguin in PenguinsType:
            if penguin.value[1] == value:
                return penguin
        raise ValueError(f"No penguin found for value: {value}")