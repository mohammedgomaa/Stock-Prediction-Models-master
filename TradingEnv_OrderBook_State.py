# -*- coding: utf-8 -*-

#import gym
import json
#import math
import numpy as np 
#from gym.utils import seeding
#from gym import error, spaces, utils
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
from concurrent import futures


class TradingEnv:
        metadata = {'render.modes': ['human']}
        def __init__(self , market_data):
            self.Index                             = 0
            self.Cash                              = [0]
            self.Coin                              = [0]
            self.Gain                              = [0]
            self.Reward                            = [0]
            self.AccGain                           = [0]
            self.AccReward                         = [0]
            self.MarketGain                        = [0]            
            self.Volatility                        = [0]
            self.Action                            = 0
            self.vwindow                           = 20 #volatility window
            self.NoOfActions                       = 11
            self.FillType                          = [0]
            self.Actionspace                       = np.linspace(-1, 1, num=self.NoOfActions)
            self.EffictiveRate                     = [0] # -ve for sell orders +ve for buy orders zero for no action
            self.Tradingfee                        = 0.0025
            self.Tick_Ask_Scaler                   = MinMaxScaler(feature_range=(0, 1))
            self.Tick_Bid_Scaler                   = MinMaxScaler(feature_range=(0, 1))
            self.Tick_Last_Scaler                  = MinMaxScaler(feature_range=(0, 1))
            self.OrderBook_buy_Rate_Scaler         = MinMaxScaler(feature_range=(0, 1))
            self.OrderBook_sell_Quantity_Scaler    = MinMaxScaler(feature_range=(0, 1))
            self.OrderBook_sell_Rate_Scaler        = MinMaxScaler(feature_range=(0, 1))
            self.MarketHistory_Price_Scaler        = MinMaxScaler(feature_range=(0, 1))
            self.MarketHistory_Quantity_Scaler     = MinMaxScaler(feature_range=(0, 1))
            self.OrderBook_buy_Quantity_Scaler     = MinMaxScaler(feature_range=(0, 1))
            self.Sell_Active_volum_Scaler          = MinMaxScaler(feature_range=(0, 1))
            self.Buy_Active_volum_Scaler           = MinMaxScaler(feature_range=(0, 1))
            self.Total_Active_Volum_Scaler         = MinMaxScaler(feature_range=(0, 1))
            self.Historical_Volum_Scaler           = MinMaxScaler(feature_range=(0, 1))
            self.Market_Data                       = market_data
            self.Action_log                        = []
            self.Volum_log                         = []
            self.Last_Significant_Action           = np.array([0,0,0,0,0,0]) # contains (coin , cash , Effrate , action , fillType , index)  action != 0 ,-1 to 1
            json_data                              = []
            with open(self.Market_Data) as file :
                for line in file:
                    json_data.append(json.loads(line))

            global json_data
            # convert an array of values into a dataset matrix
            Tick_Ask                   = []             
            Tick_Bid                   = []
            Tick_Last                  = []
            OrderBook_buy_Rate         = []
            OrderBook_buy_Quantity     = []
            OrderBook_sell_Quantity    = []
            OrderBook_sell_Rate        = []
            MarketHistory_Price        = []
            MarketHistory_Quantity     = []
            MarketHistory_FillType_Z   = [] #-1/1 encodecd
            MarketHistory_OrderType_Z  = []
            Sell_Active_volum          = []
            Buy_Active_volum           = []
            Total_Active_Volum         = []
            Historical_Volum           = []
            depth = 100
            #global Tick_Ask
            #global Tick_Bid
            global Tick_Last
            global OrderBook_buy_Rate
            global OrderBook_buy_Quantity
            global OrderBook_sell_Quantity
            global OrderBook_sell_Rate
            #global MarketHistory_Price
            #global MarketHistory_Quantity
            #global MarketHistory_FillType_Z
            #global MarketHistory_OrderType_Z
            #global Sell_Active_volum
            #global Buy_Active_volum
            #global Total_Active_Volum
            #global Historical_Volum
            with futures.ThreadPoolExecutor(max_workers=5) as ex:
                #print('main: starting')
                ex.submit(self._Extract_OrderBook_Buy )
                ex.submit(self._Extract_OrderBook_Sell )
                ex.submit(self._Extract_Tick  )
                #ex.submit(self._Extract_MarketHistory )






            #Clean Up
            #-------------------------#
            del json_data
            #-------------------------#  
            self.Last_index = len(Tick_Last)

            #MarketHistory_Price             = np.reshape(MarketHistory_Price, (int(len(MarketHistory_Price)/depth),depth))
            #MarketHistory_Quantity          = np.reshape(MarketHistory_Quantity, (int(len(MarketHistory_Quantity)/depth),depth))
            #MarketHistory_FillType_Z        = np.reshape(MarketHistory_FillType_Z, (int(len(MarketHistory_FillType_Z)/depth),depth))
            #MarketHistory_OrderType_Z       = np.reshape(MarketHistory_OrderType_Z, (int(len(MarketHistory_OrderType_Z)/depth),depth))
            OrderBook_buy_Quantity          = np.reshape(OrderBook_buy_Quantity, (int(len(OrderBook_buy_Quantity)/depth),depth))
            OrderBook_buy_Rate              = np.reshape(OrderBook_buy_Rate, (int(len(OrderBook_buy_Rate)/depth),depth))
            OrderBook_sell_Quantity         = np.reshape(OrderBook_sell_Quantity,(int(len(OrderBook_sell_Quantity)/depth),depth))
            OrderBook_sell_Rate             = np.reshape(OrderBook_sell_Rate,(int(len(OrderBook_sell_Rate)/depth),depth))
            #Buy_Active_volum                = np.reshape(Buy_Active_volum ,(1,len(Buy_Active_volum)))
            #Sell_Active_volum               = np.reshape(Sell_Active_volum ,(1,len(Sell_Active_volum)))
            #Historical_Volum                = np.reshape(Historical_Volum ,(1,len(Historical_Volum)))
            #Total_Active_Volum              = Buy_Active_volum + Sell_Active_volum
            #Tick_Ask                        = np.reshape(Tick_Ask ,(1,len(Tick_Ask)))
            #Tick_Bid                        = np.reshape(Tick_Bid ,(1,len(Tick_Bid)))
            Tick_Last                       = np.reshape(Tick_Last ,(1,len(Tick_Last)))
            
            #Tick_Ask_Norm                   = self.Tick_Ask_Scaler.fit_transform( Tick_Ask.T ).T #we rotat the  input to normalize over the time aixs and re rotate it to be sutale for concatination in the state matrix
            #Tick_Bid_Norm                   = self.Tick_Bid_Scaler.fit_transform( Tick_Bid.T ).T
            #Tick_Last_Norm                  = self.Tick_Last_Scaler.fit_transform(  Tick_Last.T ).T
            OrderBook_buy_Rate_Norm         = self.OrderBook_buy_Rate_Scaler.fit_transform(  OrderBook_buy_Rate )
            OrderBook_sell_Quantity_Norm    = self.OrderBook_sell_Quantity_Scaler.fit_transform(  OrderBook_sell_Quantity )
            OrderBook_sell_Rate_Norm        = self.OrderBook_sell_Rate_Scaler.fit_transform(  OrderBook_sell_Rate )
            #MarketHistory_Price_Norm        = self.MarketHistory_Price_Scaler.fit_transform(  MarketHistory_Price )
            #MarketHistory_Quantity_Norm     = self.MarketHistory_Quantity_Scaler.fit_transform(  MarketHistory_Quantity )
            OrderBook_buy_Quantity_Norm     = self.OrderBook_buy_Quantity_Scaler.fit_transform(  OrderBook_buy_Quantity )
            #Sell_Active_volum_Norm          = self.Sell_Active_volum_Scaler.fit_transform(  Sell_Active_volum.T ).T
            #Buy_Active_volum_Norm           = self.Buy_Active_volum_Scaler.fit_transform(  Buy_Active_volum.T ).T
            #Total_Active_Volum_Norm         = self.Total_Active_Volum_Scaler.fit_transform(  Total_Active_Volum.T ).T
            #Historical_Volum_Norm           = self.Historical_Volum_Scaler.fit_transform(  Historical_Volum.T ).T
            
            
            
            #Tick_Ask_Norm                   = self._repeatN(Tick_Ask_Norm,100)
            #Tick_Bid_Norm                   = self._repeatN(Tick_Bid_Norm,100)
            #Tick_Last_Norm                  = self._repeatN(Tick_Last_Norm,100)
            #Buy_Active_volum_Norm           = self._repeatN(Buy_Active_volum_Norm,100)
            #Sell_Active_volum_Norm          = self._repeatN(Sell_Active_volum_Norm,100)
            #Historical_Volum_Norm           = self._repeatN(Historical_Volum_Norm,100)
            #Total_Active_Volum_Norm         = self._repeatN(Total_Active_Volum_Norm,100)

            
            self.OrderBook_buy_Rate         = OrderBook_buy_Rate
            self.OrderBook_sell_Quantity    = OrderBook_sell_Quantity
            self.OrderBook_sell_Rate        = OrderBook_sell_Rate
            self.OrderBook_buy_Quantity     = OrderBook_buy_Quantity
            self.Tick_Last                  = Tick_Last.T

            '''
            self.Tick_Ask1                  = Tick_Ask
            self.Tick_Ask                   = Tick_Ask_Norm             
            self.Tick_Bid                   = Tick_Bid_Norm
            self.Tick_Last                  = Tick_Last_Norm
            self.MarketHistory_Price        = MarketHistory_Price_Norm
            self.MarketHistory_Quantity     = MarketHistory_Quantity_Norm
            self.MarketHistory_FillType_Z   = MarketHistory_FillType_Z #ZERO encodecd
            self.MarketHistory_OrderType_Z  = MarketHistory_OrderType_Z
            self.Sell_Active_volum          = Sell_Active_volum_Norm
            self.Buy_Active_volum           = Buy_Active_volum_Norm
            self.Total_Active_Volum         = Total_Active_Volum_Norm
            self.Historical_Volum           = Historical_Volum_Norm
            '''
            Input_Feturs      = np.concatenate((
                                   #Tick_Ask_Norm,
                                   #Tick_Bid_Norm,
                                   #Tick_Last_Norm,
                                   OrderBook_buy_Rate_Norm,
                                   OrderBook_buy_Quantity_Norm,
                                   OrderBook_sell_Quantity_Norm,
                                   OrderBook_sell_Rate_Norm
                                   #MarketHistory_Price_Norm,
                                   #MarketHistory_Quantity_Norm,
                                   #MarketHistory_FillType_Z,
                                   #MarketHistory_OrderType_Z,
                                   #Buy_Active_volum_Norm,
                                   #Sell_Active_volum_Norm,
                                   #Historical_Volum_Norm,
                                   #Total_Active_Volum_Norm

                                  ),axis = 1)
            self.EnvStats      = np.reshape(Input_Feturs,[Input_Feturs.shape[0],int(Input_Feturs.shape[1]/20),20])
            
            #Clean Up
            #----------------------------------#
            #del MarketHistory_Price
            #del MarketHistory_Quantity
            #del MarketHistory_FillType_Z  #-1/1 encodecd
            #del MarketHistory_OrderType_Z
            #del OrderBook_buy_Quantity 
            #del OrderBook_buy_Rate 
            #del OrderBook_sell_Quantity
            #del OrderBook_sell_Rate 
            #del Tick_Ask
            #del Tick_Bid
            #del Tick_Last
            #del Sell_Active_volum
            #del Buy_Active_volum
            #del Total_Active_Volum
            #del Historical_Volum
            del Input_Feturs
            #del MarketHistory_Price_Norm
            #del MarketHistory_Quantity_Norm
            del OrderBook_buy_Quantity_Norm
            del OrderBook_buy_Rate_Norm 
            del OrderBook_sell_Quantity_Norm
            del OrderBook_sell_Rate_Norm 
            #del Tick_Ask_Norm
            #del Tick_Bid_Norm
            #del Tick_Last_Norm
            #del Sell_Active_volum_Norm
            #del Buy_Active_volum_Norm
            #del Total_Active_Volum_Norm
            #del Historical_Volum_Norm
           #--------------------------------------#


        def _Extract_OrderBook_Sell(self):
            global OrderBook_sell_Rate
            global OrderBook_sell_Quantity
            #print('sell: starting')
            global json_data
            for i in range(len(json_data)):

                if ((json_data[i] != [] or None) and
                        (json_data[i]['Tick'] != [] or None) and
                        (json_data[i]['OrderBook'] != [] or None) and
                        (json_data[i]['MarketHistory'] != [] or None) and
                        (json_data[i]['Tick']['result'] is not None) and
                        (json_data[i]['OrderBook']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['buy'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['sell'] != [] or None) and
                        (json_data[i]['Tick']['result'] != [] or None) and
                        (json_data[i]['Tick']['result']['Ask'] != [] or None) and
                        (json_data[i]['Tick']['result']['Bid'] != [] or None) and
                        (json_data[i]['Tick']['result']['Last'] != [] or None)
                ):


                    for m in range(len((json_data[1]['OrderBook']['result']['sell']))):
                        OrderBook_sell_Quantity.append(json_data[i]['OrderBook']['result']['sell'][m]['Quantity'])
                        OrderBook_sell_Rate.append(json_data[i]['OrderBook']['result']['sell'][m]['Rate'])
                    #Sell_Active_volum.append(sum(OrderBook_sell_Rate))


        def _Extract_OrderBook_Buy(self):
            #print('buy: starting')
            global OrderBook_buy_Rate
            global OrderBook_buy_Quantity
            global json_data
            for i in range(len(json_data)):

                if ((json_data[i] != [] or None) and
                        (json_data[i]['Tick'] != [] or None) and
                        (json_data[i]['OrderBook'] != [] or None) and
                        (json_data[i]['MarketHistory'] != [] or None) and
                        (json_data[i]['Tick']['result'] is not None) and
                        (json_data[i]['OrderBook']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['buy'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['sell'] != [] or None) and
                        (json_data[i]['Tick']['result'] != [] or None) and
                        (json_data[i]['Tick']['result']['Ask'] != [] or None) and
                        (json_data[i]['Tick']['result']['Bid'] != [] or None) and
                        (json_data[i]['Tick']['result']['Last'] != [] or None)
                ):

                    for m in range(len((json_data[1]['OrderBook']['result']['buy']))):
                        OrderBook_buy_Quantity.append(json_data[i]['OrderBook']['result']['buy'][m]['Quantity'])
                        OrderBook_buy_Rate.append(json_data[i]['OrderBook']['result']['buy'][m]['Rate'])
                    #Buy_Active_volum.append(sum(OrderBook_buy_Rate))

        def _Extract_MarketHistory(self):
            #print('history: starting')
            global MarketHistory_Price
            global MarketHistory_Quantity
            global MarketHistory_FillType_Z
            global MarketHistory_OrderType_Z
            global MarketHistory_OrderType_Z
            global Sell_Active_volum
            global Buy_Active_volum
            global Total_Active_Volum
            global Historical_Volum
            global json_data
            for i in range(len(json_data)):

                if ((json_data[i] != [] or None) and
                        (json_data[i]['Tick'] != [] or None) and
                        (json_data[i]['OrderBook'] != [] or None) and
                        (json_data[i]['MarketHistory'] != [] or None) and
                        (json_data[i]['Tick']['result'] is not None) and
                        (json_data[i]['OrderBook']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['buy'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['sell'] != [] or None) and
                        (json_data[i]['Tick']['result'] != [] or None) and
                        (json_data[i]['Tick']['result']['Ask'] != [] or None) and
                        (json_data[i]['Tick']['result']['Bid'] != [] or None) and
                        (json_data[i]['Tick']['result']['Last'] != [] or None)
                ):


                    for m in range(len((json_data[1]['MarketHistory']['result']))):
                        MarketHistory_Price.append(json_data[i]['MarketHistory']['result'][m]['Price'])
                        MarketHistory_Quantity.append(json_data[i]['MarketHistory']['result'][m]['Quantity'])

                        if (json_data[i]['MarketHistory']['result'][m]['FillType']) == 'PARTIAL_FILL':
                            MarketHistory_FillType_Z.append(-1)
                        elif (json_data[i]['MarketHistory']['result'][m]['FillType']) == 'FILL':
                            MarketHistory_FillType_Z.append(1)
                        if (json_data[i]['MarketHistory']['result'][m]['OrderType']) == 'BUY':
                            MarketHistory_OrderType_Z.append(-1)
                        elif (json_data[i]['MarketHistory']['result'][m]['OrderType']) == 'SELL':
                            MarketHistory_OrderType_Z.append(1)
                    Historical_Volum.append(sum(MarketHistory_Quantity))

        def _Extract_Tick(self):
            #print('Tick: starting')
            global Tick_Ask
            global Tick_Bid
            global Tick_Last
            global json_data
            for i in range(len(json_data)):

                if ((json_data[i] != [] or None) and
                        (json_data[i]['Tick'] != [] or None) and
                        (json_data[i]['OrderBook'] != [] or None) and
                        (json_data[i]['MarketHistory'] != [] or None) and
                        (json_data[i]['Tick']['result'] is not None) and
                        (json_data[i]['OrderBook']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] is not None) and
                        (json_data[i]['MarketHistory']['result'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['buy'] != [] or None) and
                        (json_data[i]['OrderBook']['result']['sell'] != [] or None) and
                        (json_data[i]['Tick']['result'] != [] or None) and
                        (json_data[i]['Tick']['result']['Ask'] != [] or None) and
                        (json_data[i]['Tick']['result']['Bid'] != [] or None) and
                        (json_data[i]['Tick']['result']['Last'] != [] or None)
                ):

                    #Tick_Ask.append(json_data[i]['Tick']['result']['Ask'])
                    #Tick_Bid.append(json_data[i]['Tick']['result']['Bid'])
                    Tick_Last.append(json_data[i]['Tick']['result']['Last'])
                    #print(Tick_Last[-1])


        def _repeatN(self ,arr , n ):
                x1 = np.repeat(arr.T,n)
                x2 = np.reshape(x1,[arr.T.shape[0],n])
                return x2      
            
            
        def _ActionGain(self):
            if self.Action != 0 :
                #index = self.Index  ;
                coin = self.Last_Significant_Action[: , 0] ; cash = self.Last_Significant_Action[: , 1] ; rate = self.Last_Significant_Action[-1 , 2]
                cashgain      =  (cash[-1]-cash[-2]*(1+self.Tradingfee))/(cash[-1]+0.000000000005) #to avoid devion by zero small nuber added
                if abs(cashgain) >10 :
                    cashgain = 0
                coingain      =  (coin[-1]-coin[-2]*(1+self.Tradingfee))/(coin[-1]+0.000000000005)
                if abs(coingain) > 10:
                    coingain = 0
                cashcoingain  = ((cash[-1]-cash[-2]*(1+self.Tradingfee)) + (coin[-1]-coin[-2]*(1+self.Tradingfee)) * rate) / (cash[-2] + coin[-2] * rate + 0.0000000005)
                if abs(cashcoingain) > 10:
                    cashcoingain = 0
                totalGain     =  2*cashcoingain + 0.5 * cashgain + 0.5 * coingain

            else:
                totalGain = 0
            self.Gain.append(totalGain)
            self.AccGain.append(self.AccGain[-1] + totalGain)
            return  totalGain
        
        def _GetMarketGain(self):
            if self.Action !=0 :
                last_action_Index = int(self.Last_Significant_Action[-2,5])
            else :
                last_action_Index =int(self.Last_Significant_Action[-1,5])

            currindex = self.Index ; rate = self.Tick_Last[:] # shouled be calculated from the original data
            MarketGain = ((rate[currindex]-rate[last_action_Index])/rate[currindex])
            self.MarketGain.append(MarketGain)
            return MarketGain 
        
       
        def _GetVolatility(self):
            #calculat volatility
            LocalVolatility = 0
            windw           = self.vwindow  #volatility window
            #TickPriceIndex  = 2  #index of Tick_Last in EnvStats matrix # edited to avoid -ves in rates
            if self.Index > windw :
                LocalVolatility = np.std(self.Tick_Last[(self.Index-windw):self.Index])#(self.Index-windw):self.Index ,TickPriceIndex,0]) # usinf index 0 becouse the value is repeatimested 100  # Discarded Not Using Envstats
            else :
                LocalVolatility = np.std(self.Tick_Last[:self.Index])
            self.Volatility.append(LocalVolatility)
            return LocalVolatility
        
        
        def _CalculateReward(self):
                if self.Index != 0 :
                    ActionGain       = self._ActionGain()
                    MarketGain       = self._GetMarketGain()
                    volaility        = self._GetVolatility()
                    #bahaviorenforce = self._BehaviorAssit( ) # to be implemented to encourge or discarge behavior
                    reward           = (ActionGain-0.5*MarketGain-0.5*volaility)+0.5*self.FillType[self.Index]*abs((ActionGain-0.5*MarketGain-0.5*volaility))   # inportant To revize
                else :
                    reward        = 0.000005
                reward = reward / 1000
                self.Reward.append(reward)
                self.AccReward.append(self.AccReward[-1]+reward)
                return reward



        def _BuyCoin(self , cvol): #cvol is cash volum
                FillType = 1 #-1 is partially filled 1 is completrly filled
                if round(cvol ,5) != 0 : # to avoid near zero float error

                    # there is an uber limet on the bougt coin becouse of the orderbook depth so we shouled chek the avlability of the volum firist
                    #assert (np.sum(self.OrderBook_sell_Quantity[self.Index,:]) > vol ), "exceptin buy  avalable vol "
                    i        = -1
                    AvVol    = self.OrderBook_sell_Quantity[self.Index,:]
                    AvRate   = self.OrderBook_sell_Rate [self.Index,:]
                    xvol     = np.sum( self.OrderBook_sell_Rate [self.Index,:] * self.OrderBook_sell_Quantity[self.Index,:]) #avelable volum in this tick (curent orderbook) in cash
                    AccVol = 0
                    DistVol = []
                    DistRate = []
                    if cvol > xvol :
                        cvol = xvol
                        DistVol = AvVol
                        DistRate= AvRate
                        FillType= -1
                    else:
                        index = 0
                        while round(cvol ,5) != 0:
                            DistRate.append(AvRate[index])
                            if cvol > AvVol[index]*AvRate[index] :
                                DistVol.append(AvVol[index])
                                cvol = cvol-AvVol[index]*AvRate[index]
                            else :
                                DistVol.append(cvol/AvRate[index])
                                cvol = 0
                            index += 1
                            AccVol += (DistRate[-1]*DistVol[-1])

                    DistVol  = np.array(DistVol)
                    DistRate = np.array(DistRate)
                    effRate  = np.sum(DistVol*DistRate)/np.sum(DistVol)
                    coin     = self.Coin[self.Index] + np.sum(DistVol)
                    coin     = coin -self.Coin[self.Index -1]*self.Tradingfee #taking The trading fee
                    cash     = round(self.Cash[self.Index] - float(np.sum(DistVol * DistRate)),5)
                else :
                    coin , cash ,  effRate = self.Coin[self.Index] , self.Cash[self.Index] , 0
                return  coin , cash ,  effRate ,FillType # taking the trading fee
            

        def _SellCoin(self , vol):
            # there is an uber limet on the bougt coin becouse of the orderbook depth so we shouled chek the avlability of the volum firist
            FillType = 1 # -1 is partially filled 1 is completrly filled
            if round(vol ,5) != 0 : # to avoid near zero float error
                i = -1
                AvVol = self.OrderBook_buy_Quantity[self.Index, :]
                AvRate = self.OrderBook_buy_Rate[self.Index, :]
                AccVol = 0
                DistVol = []
                DistRate = []
                xvol = np.sum(self.OrderBook_buy_Quantity[self.Index,:]) #avelable volum in this tick (curent orderbook)
                if vol > xvol :
                    vol = xvol
                    FillType = -1
                    DistVol  = AvVol
                    DistRate = AvRate
                else:
                    while AccVol<vol:
                        i      += 1
                        AccVol =AccVol+AvVol[i]
                        DistRate.append(AvRate[i])

                    for index in range(len(DistRate)):
                        if vol>AvVol[index]:
                            DistVol.append(AvVol[index])
                            vol = vol-AvVol[index]
                        else:
                            DistVol.append(vol)

                DistVol  = np.array(DistVol)
                DistRate = np.array(DistRate)
                effRate  = np.sum(DistVol*DistRate)/np.sum(DistVol)
                coin     = round(self.Coin[self.Index] - np.sum(DistVol),5)
                cash     = self.Cash[self.Index] + (np.sum(DistVol)*effRate)
                cash     = cash-self.Cash[-1]*self.Tradingfee #taking the trading fee

              
            else :
                coin , cash ,  effRate = self.Coin[self.Index] , self.Cash[self.Index] ,0

            return  coin ,cash ,  effRate ,FillType

        
        
        

        def _TakeAction(self):
            #action logic
            # does not take in account unishial 0 cash or coin
            if   round(self.Action ,3 ) > 0 :
                 self.Volum_log.append((abs(self.Action * self.Cash[self.Index]),'cash'))
                 coin , cash , effRate , FillType  = self._BuyCoin(abs(self.Action*self.Cash[self.Index]))
                 self.Last_Significant_Action      = np.vstack((self.Last_Significant_Action,[coin, cash, effRate, self.Action, FillType, self.Index]))

            elif round(self.Action ,3) < 0 :
                 self.Volum_log.append((abs(self.Action * self.Coin[self.Index]),'coin'))
                 coin , cash , effRate , FillType   = self._SellCoin(abs(self.Action*self.Coin[self.Index]))
                 self.Last_Significant_Action       = np.vstack((self.Last_Significant_Action,[coin, cash, effRate, self.Action, FillType, self.Index]))
            else:
                 self.Volum_log.append((0,'No action '))
                 effRate = 0
                 coin     = self.Coin[self.Index]
                 cash     = self.Cash[self.Index]
                 FillType = 0
            #update cash and coin and EffictiveRate
            self.Coin.append(coin)    #adjusting for market tradfing fee
            self.Cash.append(cash)    #adjusting for market tradfing fee
            self.EffictiveRate.append(effRate)
            self.FillType.append(FillType)
                   
        
        def _GetPerformanceMetrics(self):
            #make PerformanceMetrics(
            if  self.Index > 0:
                coin      = self._repeatN(np.array([(self.Coin[self.Index]-self.Coin[self.Index-1])/self.Coin[self.Index-1]]),100)
                cash      = self._repeatN(np.array([(self.Cash[self.Index]-self.Cash[self.Index-1])/self.Cash[self.Index-1]]),100)
                gain      = self._repeatN(np.array([(self.Gain[self.Index]-self.Gain[self.Index-1])/self.Gain[self.Index-1]]),100)
                accgain   = self._repeatN(np.array([(self.AccGain[self.Index]-self.AccGain[self.Index-1])/self.AccGain[self.Index-1]]),100)
                reward    = self._repeatN(np.array([(self.Rward[self.Index]-self.Rward[self.Index-1])/self.Rward[self.Index-1]]),100)
                accreward = self._repeatN(np.array([(self.AccReward[self.Index]-self.AccReward[self.Index-1])/self.AccReward[self.Index-1]]),100)
                volatility= self._repeatN(np.array([(self.Volatility[self.Index]-self.Volatility[self.Index-1])/self.Volatility[self.Index-1]]),100)
                                   
            else:
                coin      = self._repeatN(np.array([0]),100)
                cash      = self._repeatN(np.array([1]),100)
                gain      = self._repeatN(np.array([0]),100)
                accgain   = self._repeatN(np.array([0]),100)
                reward    = self._repeatN(np.array([0]),100)
                accreward = self._repeatN(np.array([0]),100)
                volatility= self._repeatN(np.array([0]),100)
                                                      
                                     
            PerformanceMetrics      = np.concatenate((
                                                    coin,
                                                    cash,
                                                    gain,
                                                    accgain,
                                                    reward,
                                                    accreward,
                                                    volatility
                                                    ),axis = 0)
                                                
                                                    
            return PerformanceMetrics
                             

        
        def _step(self, action):
            self.Action = action
            self._TakeAction()
            reward       = self._CalculateReward()
            return reward
    
    
        def _reset(self):
            self.Index         =  0
            self.Cash          = [self.Cash[0]]
            self.Coin          = [self.Coin[0]]
            self.Gain          = [0]
            self.Reward        = [0]
            self.AccGain       = [0]
            self.AccReward     = [0]
            self.Volatility    = [0]
            self.EffictiveRate = [0]
            self.MarketGain    = [0]
            self.Action        = 0
            self.FillType      = [0]



        #interface with external codes    
        def new_episode(self):
            self._reset()
         
        
        def get_state(self) :
            '''
            perfMetrices = self._GetPerformanceMetrics()
            State = np.concatenate((
                        self.EnvStats[self.Index],
                        perfMetrices
                          ),axis = 0)
            return State
            '''
            return  self.EnvStats[self.Index]
        
        @property
        def is_episode_finished(self):
            finshed = False 
            if self.Index >= self.Last_index  or (round(self.Cash[self.Index] ,5)+round(self.Coin[self.Index] ,5))== 0:
                finshed = True
                
            return  finshed
        
        
        def make_action(self ,action): #self.actions[a]
            self.Action_log.append(self.Actionspace[action])
            r               = self._step(self.Actionspace[action]) # shouled retern reward
            self.Index     += 1
            return r
        
        def set_coin(self , coin):
            self.Coin[0] = coin
           
        def set_cash(self , cash):
            self.Cash[0] = cash 
 
