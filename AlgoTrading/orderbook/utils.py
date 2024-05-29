"""
Utilities package containing all the
assistant functions needed to generate
the orderbook. This package also
contains functions and methods for the
second part of the exercise, which
is devoted to create some alphas for
the aggregated orderbooks
"""

from typing import Tuple, List
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
n_levels = 5
price_tick = 5

# the orderbook type is a dictionary that contains
# two keys, bid and ask.
# Each bid/ask are on dictionaries on themselves
# with keys integers from 0 to n, where n is the number of distinct
# prices in that side of the orderbook. Their values
# are lists of three elements, containing the price at that level
# the aggregated volume on that level and a set containing the individual
# order ids at that level.


# TO-DO: ??? Group all the printing/outputting functions ???
# These functions are independent from the ob object on itself,
# that is why is not a function of such class

def generate_n_levels_labels() -> list[str]:
    labels = []
    for side in ['a','b']:
        for i in range(n_levels):
            labels.extend([f'{side}p{i}',f'{side}q{i}'])
    return labels


def generate_ob_derived_metrics() -> list[str]:
    return ['mid_price']

def generate_header() -> list[str]:
    header = ['timestamp','price','side']
    header.extend(generate_n_levels_labels())
    header.extend(generate_ob_derived_metrics())

    return header

class orderbook:
    # TO-DO_1: Centralize all relevant fields (price, side, action, etc)
    #          in one place
    # TO-DO_3: Create types for self.bid/self.ask
    # TO-DO_4: Add helper messages to functions to increase readability

    def __init__(self, log_warnings:bool=False) -> None:
        self.bid = {}
        self.ask = {}
        self.update : ob_update = ()
        # this dictionary keeps track of the current active
        # orders
        self.active_orders = {}
        self.log = log_warnings

    def _format_input(self, in_line:List) -> ob_update:
        ts,side,action,id,price,volume = in_line
        # By default all fields are read as strings
        # Change those that are not strings
        ts = int(ts)
        id = int(id)
        price = float(price)
        volume = int(volume)

        return ts,side,action,id,price,volume
    
    def _selector(self, side:str) -> dict:
        if side=='b':
            return self.bid
        elif side=='a':
            return self.ask
        else:
            # we should never get here, just a sanity check
            raise ValueError("side is not b or a, check!")
    
    def _add_orderid(self) -> None:
        ts,side,id,price,vol = self.update[0],self.update[1],self.update[3],self.update[4],self.update[5]
        # check if price is in the ob
        ob_side = self._selector(side)
        if price not in ob_side:
            ob_side[price] = [vol,set([id])]
        else:
            if self.log and id in ob_side[price][1]:
                warnings.warn(f"order_id={id} with timestamp={ts} was added to side={side} "
                                f"but it was already in the orderbook, seems weird!")
            # All this code should never be reached because an order that is already
            # in the orderbook should not be added for a second time. This was just
            # developed as control/sanity check
            ob_side[price][0] += vol
            ob_side[price][1].add(id)
        
        if self.log:
            print(f"Order_id={id} was added to side={side} with "
                  f"price={price} and vol={vol}")
        
        # Finally add the order to the dictionary of active orders
        self.active_orders[id] = [price,vol]         

    def _remove_orderid(self) -> None:
        side = self.update[1]
        ob_side = self._selector(side)
        id = self.update[3]
        price = self.update[4]
        vol = self.update[5]
        if price not in ob_side:
            raise ValueError(f"order_id={id} is deleted from side={side} but that price={price} "
                             f"was never in the orderbook")
        else:
            if id not in ob_side[price][1]:
                raise ValueError(f"price={price} is present in the ob but that "
                                 f"order_id={id} is not, CHECK!")
            else:
                ob_side[price][0] -= vol
                ob_side[price][1].remove(id)
                
                # Check that volumes are never negative
                if ob_side[price][0]<0:
                    raise ValueError(f"side={side}, price={price} has negative volume, vol={ob_side[price][0]}")
                
                if ob_side[price][0]==0:
                    # we should not have any order_id here
                    if ob_side[price][1]:
                        raise ValueError(f"there is no volume in this side={side}, price={price} "
                                         f"but there are some order_ids={ob_side[price][1]}")
                    else:
                        del ob_side[price]
                        if self.log:
                            warnings.warn(f"removing price={price} from orderbook")
                
        # Finally remove the order from the dictionary of active orders
        del self.active_orders[id]

    def _modify_orderid(self) -> None:
        id = self.update[3]
        if id not in self.active_orders:
            raise ValueError(f"order_id={id} wants to be modified but "
                             f"it's not in the list of active orders, CHECK!!")
        new_price,new_vol = self.update[4],self.update[5]
        old_price,old_vol = self.active_orders[id]
        side = self.update[1]
        ob_side = self._selector(side)
        if new_price==old_price and new_vol!=old_vol:
            # in this case we only need to modify the volume
            if self.log:
                warnings.warn(f"new_vol={new_vol}, old_vol={old_vol}, "
                            f"level_vol += {new_vol-old_vol}")
            ob_side[old_price][0] += new_vol-old_vol
            # Check that volumes are never negative
            if ob_side[old_price][0]<0:
                raise ValueError(f"side={side}, price={old_price} has negative " 
                                 f"volume, vol={ob_side[old_price][0]}")
            # only change the volume as price is the same
            self.active_orders[id][1] = new_vol

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
            
            # only change the price as the volume is the same
            self.active_orders[id][0] = new_price

        else:
            raise ValueError("Price and volume have changed at the same time, check!")
    
    def process_update(self, update:list) -> None:
        self.update = self._format_input(update)
        action = self.update[2]

        match action:
            case 'a':
                # add_price_vol
                self._add_orderid()
            case 'd':
                # go to the specific level where the order sits and removes it
                self._remove_orderid()
            case 'm':
                # in case of a modification, according to the instructions
                # either price or vol have changed, but not both.
                self._modify_orderid()
    
    #TO-D0: Develop or remove this
    def _quality_checks(self) -> None:
        return None
    
    def _generate_statistics(self) -> None:

        # bid/ask prices can be None though, perfome checks:
        if self.ob_view['ap0'] and self.ob_view['bp0']:
            # mid_price => 0.5*(best_ask+best_bid)
            self.ob_view['mid_price'] = 0.5*(self.ob_view['ap0']+self.ob_view['bp0'])
        else:
            self.ob_view['mid_price'] = None
    
    def generate_ob_view(self) -> dict:

        self.ob_view = {'timestamp':self.update[0],
                        'price':self.update[4],
                        'side':self.update[1]}

        for side in ['a','b']:
            ob_side = self._selector(side)
            # Since we have to retrieve prices in sorted order,
            # one quick way to do it is via a heap
            # Note that heaps are min heaps in Python, so to get
            # a max heap(for the bid side) we need to insert negative prices,
            # which are of course converted again afterwards before being
            # distributed.
            price_vols = [[price,val[0]] if side=='a' else [-1*price,val[0]] for price,val in ob_side.items()]
            heapify(price_vols)
            
            for i in range(n_levels):
                if price_vols:
                    price,vol = heappop(price_vols)
                    self.ob_view[f"{side}p{i}"] = price if price>0 else -1*price
                    self.ob_view[f"{side}q{i}"] = vol
                else:
                    self.ob_view[f"{side}p{i}"] = None
                    self.ob_view[f"{side}q{i}"] = 0

        #TO-DO: Perform some quality checks to ensure the produced
        #       orderbook has the desired properties
        #self._quality_checks()

        # Add orderbook derived statistics to the view
        self._generate_statistics()

        if self.log:
            print(f"order book view => {self.ob_view}")
        
        return self.ob_view
    
    def format_output(self) -> list:
        values = [self.ob_view[label] for label in generate_header()]
        return values