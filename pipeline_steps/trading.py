from backtesting import Strategy


class MyStrategy(Strategy):
    predicted_column_name = 'DIFF_1'
    buying_amount = 1

    trusted_difference = 0.1 # we will trust predicted_prices, that are within a 10% margin of error
    # PROBLEM: the bigger the price the bigger the trusted interval!


    def init(self):
        pass


    def can_buy(self, current_price, predicted_price):
        min_trusted_price = current_price * (1-self.trusted_difference)
        max_trusted_price = current_price * (1+self.trusted_difference)
        if min_trusted_price < predicted_price and predicted_price < max_trusted_price:
            # if the model is predicting a positive change
            if predicted_price > current_price:
                return True
        return False
       
        
    def can_sell(self, current_price, predicted_price):
        min_trusted_price = current_price * (1-self.trusted_difference)
        max_trusted_price = current_price * (1+self.trusted_difference)
        if min_trusted_price < predicted_price and predicted_price < max_trusted_price:
            # if the model is predicting a negative change
            if predicted_price < current_price:
                return True
        return False
    
    
    def next(self):
        current_price = self.data.Close[-1]
        predicted_price = current_price + self.data[f'PRED_{self.predicted_column_name}'][-1]
        
        if self.can_buy(current_price, predicted_price):
            self.buy(size=self.buying_amount)
        
        if self.can_sell(current_price, predicted_price):
            self.position.close()