"""
Utilities package containing all the
assistant functions needed to generate
the orderbook. This package also
contains functions and methods for the
second part of the exercise, which
is devoted to create some alphas for
the aggregated orderbooks
"""

from typing import Tuple, Sequence
import warnings
from heapq import heapify,heappop

# Why I am using a tuple instead of a list?
# We do not need to modify the incoming order book
# updates, so I've choosen it as a immutable data type
ob_update = Tuple[int,#timestamp
                  str,#side
                  str,#action
                  int,#id
                  float,#price. From the data seems like lot size is 5 monetary
                  # units, but in general it could be any floating point number
                  int, #quantity, number of shares/contracts
                  ]




# what info I need to output?
# timestamp of original update
# price of original update
# side of original update
# orderbook, containing 5 levels for bid/ask respectively

# the orderbook type is a dictionary that contains
# two keys, bid and ask.
# Each bid/ask are on dictionaries on themselves
# with keys integers from 0 to n, where n is the number of distinct
# prices in that side of the orderbook. Their values
# are lists of three elements, containing the price at that level
# the aggregated volume on that level and a set containing the individual
# order ids at that level.

#ob_type = Dict[str,Dict[int,Sequence[float,int,set]]]

class orderbook:
    def __init__(self) -> None:
        self.bid = {}
        self.ask = {}
        self.update : ob_update = ()
        # this dictionary keeps track of the current active
        # orders
        self.active_orders = {}
    
    def _selector(self, side:str) -> dict:
        if side=='b':
            return self.bid
        elif side=='a':
            return self.ask
        else:
            # we should never get here
            raise ValueError("side is not b or a, check!")
    
    def _add_orderid(self, id:str) -> None:
        ts,side,price,qty = self.update[0],self.update[1],self.update[4],self.update[5]
        # check if price is in the ob
        ob_side = self._selector(side)
        if price not in ob_side:
            ob_side[price] = [qty,set(id)]
        else:
            warnings.warn(f"order_id={id} with timestamp={ts} was added to side={side} "
                          f"but it was already in the orderbook, seems weird!")
            # ALL THIS CODE ABOVE SHOULD BE NEVER REACHED BECAUSE AN ORDER THAT IS ALREADY IN
            # THE ORDERBOOK SHOULD NOT BE ADDED FOR A SECOND TIME.
            ob_side[price][0] += qty
            ob_side[price][1].add(id)
            
        print(f"Order_id={id} was added to side={side} with"
              f"price={price} and quantity={qty}")
        # finally add the order to the dictionary of active orders
        self.active_orders[id] = [price,qty]         

    def _remove_orderid(self, id:str) -> None:
        side = self.update[1]
        ob_side = self._selector(side)
        price = self.update[4]
        vol = self.update[5]
        if price not in ob_side[price]:
            raise ValueError(f"order_id={id} is deleted but that price={price} "
                             f"was never in the orderbook")
        else:
            if id not in ob_side[price][1]:
                raise ValueError(f"price={price} is present in the ob but that "
                                 f"order_id={id} is not, CHECK!")
            else:
                ob_side[price][0] -= vol
                ob_side[price][1].remove(id)
                if ob_side[price][0]==0:
                    # we should not have any order_id here
                    if ob_side[price][1]:
                        raise ValueError(f"there is no volume in this price={price} level "
                                         f"but there are some order_ids={ob_side[price][1]}")
                    else:
                        del ob_side[price]
                        warnings.warn(f"removing price={price} from orderbook")
        
        # Finally remove the order from the dictionary of active orders
        del self.active_orders[id]

    def _modify_orderid(self, id:str, new_price:float, new_vol:int) -> None:
        if id not in self.active_orders:
            raise ValueError(f"order_id={id} wants to be modified but "
                             f"it's not in the list of active orders, CHECK!!")
        old_price,old_vol = self.active_orders[id]
        side = self.update[1]
        ob_side = self._selector(side)
        if new_price==old_price and new_vol!=old_vol:
            # in this case we only need to modify the volume
            print(f"new_vol={new_vol}, old_vol={old_vol}, "
                  f"level_vol += {new_vol-old_vol}")
            ob_side[old_price][0] += new_vol-old_vol
        elif new_vol==old_vol and new_price!=old_price:
            # volume is the same, but price has changed
            # In that case we need to remove the volume from
            # the old_price
            ob_side[old_price][0] -= new_vol
            ob_side[old_price][1].remove(id)

            # And add volume to the new price
            if new_price in ob_side[new_price]:
                ob_side[new_price][0] += new_vol
            else:
                ob_side[new_price] = [new_vol,set(id)]
            
            self.active_orders[id] = [new_price,new_vol]

        else:
            raise ValueError("Price and volume have changed at the same time, check!")

    
    def process_update(self,update:str):# This should return the output that needs to be written to the csv
        self.update = tuple(update.split(','))
        action = self.update[2]
        id = self.update[3]
        price = self.update[4]
        qty = self.update[5]

        match action:
            case 'a':
                # add_price_vol
                self._add_orderid(id)
            case 'd':
                # go to the specific level where the order sits and removes it
                self._remove_orderid(id)
            case 'm':
                # in case of a modification, according to the instructions
                # either price or qty have changed, but not both.
                self._modify_orderid(id,price,qty)
    
    def generate_ob_view(self) -> dict:

        # Since we have to retrieve prices in sorted order,
        # one quick way to do it is via a heap

        ob_view = {}

        for side in ['a','b']:
            ob_side = self._selector(side)
            price_vols = heapify([[price,val[0]] if ob_side=='a' else [-1*price,val[0]] for price,val in ob_side.items()])
            i = 0
            # this is limited to 5 as the instructions mention that we should only retrieve
            # the first 5 levels of the orderbook
            while i<5:
                if price_vols:
                    price,qty = heappop(price_vols)
                    ob_view[f"{side}p{i}"] = price if price>0 else -1*price
                    ob_view[f"{side}q{i}"] = qty
                else:
                    ob_view[f"{side}q{i}"] = 0
                i += 1

            print(f"order book view => {ob_view}")



# TO-DO: Start testing the code on a small sample of updates and check for bugs!