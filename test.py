
import os
from dotenv import load_dotenv
load_dotenv()
print('ALPACA_API_KEY:', os.getenv('ALPACA_API_KEY'))

