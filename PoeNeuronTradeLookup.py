# trade lookup

import sys
import os
poa_path = os.path.join(os.getcwd()) + '/../Path-of-Accounting'
print(poa_path)
sys.path.append(poa_path)
from utils.parse import price_item
import math
import pyperclip

def get_clipboard():
    return pyperclip.paste()

cache={}
def get_price_for_text(text):
    text_lines = text.split('\n')
    item_name = text_lines[2]
    if item_name in cache:
        return cache[item_name]

    results = price_item(text)
    prices = 0
    count = 0
    best_currency = 'Chaos'
    for key in results:
        item = results[key]
        type = key.split(' ')[1]
        if type != best_currency:
            prices = 0
            count = 0
        prices = prices + int(key.split(' ')[0])
        count = count + 1
        if count > 4:
            break
    if count == 0:
        return '99 Exalt'
    amount = math.ceil(prices / count)
    price = '{} {}'.format(amount, best_currency)
    cache[item_name] = price
    return price
    
def get_price_for_mouse_position():
    #PoeNeuronControls.press_ctrl_c()
    text = get_clipboard()
    price = get_price_for_text(text)
    return price
    
    
input="""
Rarity: Rare
Vengeance Ward
Lion Pelt
--------
Quality: +7% (augmented)
Evasion Rating: 707 (augmented)
--------
Requirements:
Level: 70
Dex: 150
--------
Sockets: G R 
--------
Item Level: 71
--------
Power Siphon fires an additional Projectile
--------
+30 to Intelligence
10% increased Global Accuracy Rating
79% increased Evasion Rating
+14% to Fire Resistance
11% increased Stun and Block Recovery
Reflects 5 Physical Damage to Melee Attackers
5% increased Light Radius
--------
Note: ~price 20 chaos
"""

if __name__ == "__main__":
    text=input
    price = get_price_for_text(text)
    print(price)