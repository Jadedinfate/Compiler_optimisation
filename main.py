import sys
from src.exception.exception import exception_handling

if __name__=="__main__":
    try:
        pass
        
    except Exception as e:
        raise exception_handling(e,sys)