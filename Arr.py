# import matplotlib.pyplot as plt
# import numpy as np
# import pywt
# import pandas as pd
# from utils import function as func


# def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
#     coeffs = pywt.wavedec(X, dwt_transform, level=dlevels)  # wavelet transform 'bior4.4'
#     for ca in range(0, cutoff_low):
#         coeffs[ca] = np.multiply(coeffs[ca], [0.0])

#     for ca in range(cutoff_high, len(coeffs)):
#         coeffs[ca] = np.multiply(coeffs[ca], [0.0])
#     Y = pywt.waverec(coeffs, dwt_transform)  # inverse wavelet transform
#     return Y


   
# class Arrhythmia_classification:
#     def __init__(self): 
#         self.result = ""

#     def classify_data(self, RHYTHM, lead):
        
#         """
#         Parameters:
#             :param RHYTHM: result Data
#             :param lead: name of lead
#             :return: result
#         """
#         Heart_Rate = RHYTHM[lead]["Heart_Rate"]["V"]
#         RR_Rythm = RHYTHM[lead]["RR_Rhythm"]["V"]
#         QRS_Duration = RHYTHM[lead]["QRS_Duration"]["V"]
#         PR_interval = RHYTHM[lead]["PR_Interval"]["V"]
#         RP_interval = RHYTHM[lead]["RP_Interval"]["V"]
#         Atrial_rate = RHYTHM[lead]["Atrial_rate"]["V"]
#         ventricular_rate = RHYTHM[lead]["Ventricular_rate"]["V"]
#         p_amp = RHYTHM[lead]["P_Amplitude"]["V"]
#         q_amp = RHYTHM[lead]["Q_Amplitude"]["V"] 
#         # PRinterval = RHYTHM[lead]["Pr_type"]["V"]
#         AV_relationship = RHYTHM[lead]["AV_relationship"]["V"]
#         R_peak_time = RHYTHM[lead]["R_peak_time"]["V"] 
#         # P_Duration = RHYTHM[lead]["P_Duration"]["V"] 
#         P_Duration = 49
#         # VAT = RHYTHM[lead]["VAT"]["V"]
#         VAT = 49
#         RR_Interval = RHYTHM[lead]["RR_Interval"]["V"]
#         if p_amp >= 0.1:
#             P_wave = "True"
#         else:
#             P_wave = "False"
#         if  q_amp >= -0.1:
#                 Q_wave = "True"
#         else:
#                 Q_wave = "False"   
   
#         # print(f"Rhythm: {RR_Rythm}, Heart Rate: {Heart_Rate}")

#         if 60 < Heart_Rate < 100 and RR_Rythm == "Regular":
#             self.result = "Normal sinus Rhythm"
#         if Heart_Rate <= 60  and RR_Rythm == "Regular":
#           self.result = "Sinus Bradycardia"
#         if Heart_Rate <= 70 and  PR_interval<=140:
#             self.result = "Junctional Rhythm"
#         if Heart_Rate > 70 and  PR_interval<=140 :
#             self.result = "Rapid Junctional Rhythm" 
#         if Heart_Rate <= 70 and RR_Rythm =="Irregular":
#             self.result = "Sinus Arrhythmia"  
#         if Heart_Rate <=70 and PR_interval > 140 and P_Duration <120 and p_amp <0.25:
#              self.result = "Atrial Rhythm " 
#         # if Heart_Rate > 70 and PR_interval > 140:
#         #      self.result = "Rapid Atrial Rhythm "      
#         if Heart_Rate > 100 and QRS_Duration <= 120:  
#             if RR_Rythm == "Regular" and P_wave == "True" :
#                 self.result = "Sinus Tachycardia"   
#             if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate > ventricular_rate and 200 < Atrial_rate < 350:
#                 self.result = "Atrial flutter" 
#             if  RR_Rythm == "Irregular" and 75 < ventricular_rate < 150:
#                 self.result = "Atrial flutter with variable AV condition"  
#             if RR_Rythm == "Regular" and P_wave == "False":
#                 self.result = "P wave is not present__AVNRT"    
#             if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate < ventricular_rate and RP_interval < PR_interval and RP_interval < 70:
#                 self.result = "AVNRT"
#             if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate < ventricular_rate and RP_interval > PR_interval:
#                 self.result = "Atypical AVNRT"          
#         if Heart_Rate > 100 and QRS_Duration > 120 :
           
#            if RR_Rythm == "Regular" and\
#               AV_relationship != 1 and ventricular_rate > Atrial_rate:
#                 self.result = "Ventricular_Tachycardia"
#            elif RR_Rythm == "Regular" and AV_relationship == 1 and R_peak_time >= 50:
#                 self.result = "Ventricular Tachycardia"
           
#            if RR_Rythm == "Regular" and AV_relationship != 1 and ventricular_rate < Atrial_rate :
#                 self.result = "Atrial flutter with variable AV condition"
#            elif  Heart_Rate > 100 and QRS_Duration > 120 and\
#                     RR_Rythm== "Regular" and  AV_relationship!= 1 and\
#                         ventricular_rate < Atrial_rate and\
#                         200 < Atrial_rate < 350 and QRS_Duration >= 120:
#                        self.result("Atrial flutter with aberrant conduction ,or ventricular premature complexes ")
#            elif RR_Rythm == "Irregular" and  Atrial_rate/ventricular_rate < 0.4 and\
#                 75 < ventricular_rate < 150:
#                      self.result = "Atrial flutter"      
#            elif RR_Rythm == "Irregular":
#                 self.result = "PVC  "   
#         if 50 <= Heart_Rate <= 100 and RR_Rythm == "Irregular" and P_wave == "False" and  QRS_Duration <= 120:
#             self.result = "Atrial fibrillation"
               
#         elif 100 <= Heart_Rate and RR_Rythm == "Irregular" and P_wave == "False":
#             self.result = "Atrial fibrillation with rapid ventricular response "
#         elif 50 > Heart_Rate and RR_Rythm == "Irregular" and P_wave == "False":
#             self.result = "Atrial fibrillation with slow ventricular response "
#         elif 50 >  Heart_Rate and RR_Rythm == "Irregular" and  P_wave == "False" and QRS_Duration>= 120 :
#             self.result(" Atrial fibrillation with slow ventricular response  with aberrant conduction ,or ventricular premature complexes")       
#         elif 50 <=  Heart_Rate < 100 and RR_Rythm == "Irregular" and\
#             P_wave == "False" and QRS_Duration>= 120:
#             self.result=(" Atrial fibrillation with aberrant conduction,or ventricular premature complexes") 
                     
#         else:
#             self.result += "\t" + ""
     
#         return self.result

# if __name__ == "__main__":
#     print("Hello from Arr")





# import matplotlib.pyplot as plt
# import numpy as np
# import pywt
# import pandas as pd
# from utils import function as func


# def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
#     coeffs = pywt.wavedec(X, dwt_transform, level=dlevels)  # wavelet transform 'bior4.4'
#     for ca in range(0, cutoff_low):
#         coeffs[ca] = np.multiply(coeffs[ca], [0.0])

#     for ca in range(cutoff_high, len(coeffs)):
#         coeffs[ca] = np.multiply(coeffs[ca], [0.0])
#     Y = pywt.waverec(coeffs, dwt_transform)  # inverse wavelet transform
#     return Y


   
# class Arrhythmia_classification:
#     def __init__(self): 
#         self.result = ""

#     def classify_data(self, RHYTHM, lead):
        
#         """
#         Parameters:
#             :param RHYTHM: result Data
#             :param lead: name of lead
#             :return: result
#         """
#         Heart_Rate = RHYTHM[lead]["Heart_Rate"]["V"]
#         RR_Rythm = RHYTHM[lead]["RR_Rhythm"]["V"]
#         QRS_Duration = RHYTHM[lead]["QRS_Duration"]["V"]
#         PR_interval = RHYTHM[lead]["PR_Interval"]["V"]
#         RP_interval = RHYTHM[lead]["RP_Interval"]["V"]
#         Atrial_rate = RHYTHM[lead]["Atrial_rate"]["V"]
#         ventricular_rate = RHYTHM[lead]["Ventricular_rate"]["V"]
#         p_amp = RHYTHM[lead]["P_Amplitude"]["V"]
#         q_amp = RHYTHM[lead]["Q_Amplitude"]["V"] 
#         # PRinterval = RHYTHM[lead]["PR_Interval"]["V"]
#         AV_relationship = RHYTHM[lead]["AV_relationship"]["V"]
#         R_peak_time = RHYTHM[lead]["R_peak_time"]["V"] 
#         RR_Deviation = RHYTHM[lead]["RR_Deviation"]["V"] 
#         P_Duration = 49
#         # VAT = RHYTHM[lead]["VAT"]["V"]
#         VAT = 49
#         RR_Interval = RHYTHM[lead]["RR_Interval"]["V"]
#         if p_amp >= 0.1:
#             P_wave = "True"
#         else:
#             P_wave = "False"
#         if  q_amp >= -0.1:
#                 Q_wave = "True"
#         else:
#                 Q_wave = "False"   
   
#         # print(f"Rhythm: {RR_Rythm}, Heart Rate: {Heart_Rate}")

#         if 60 < Heart_Rate < 100 and RR_Rythm == "Regular":
#             self.result = "Normal sinus Rhythm"
#         if Heart_Rate <= 60  and RR_Rythm == "Regular":
#           self.result = "Sinus Bradycardia"
#         if Heart_Rate <= 70 and  PR_interval<=140:
#             self.result = "Junctional Rhythm"
#         if Heart_Rate > 70 and  PR_interval<=140 :
#             self.result = "Rapid Junctional Rhythm" 
#         if Heart_Rate <= 70 and\
#                 RR_Deviation > 0.2*RR_Interval:
#                     self.result = "Sinus Arrhythmia"  
#         if RR_Deviation> 0.4* RR_Interval: 
#                     self.result = "Marked Sinus Arrhythmia"  
#         if Heart_Rate <=70 and  PR_interval > 140:
#                     self.result = "Atrial Rhythm " 
         
#         # if Heart_Rate > 70 and PR_interval > 140:
#         #      self.result = "Rapid Atrial Rhythm "      
#         if Heart_Rate > 100 and QRS_Duration <= 120:  
#             if RR_Rythm == "Regular" and P_wave == "True" :
#                 self.result = "Sinus Tachycardia"   
#             if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate > ventricular_rate and 200 < Atrial_rate < 350:
#                 self.result = "Atrial flutter" 
#             if  RR_Rythm == "Irregular" and 75 < ventricular_rate < 150:
#                 self.result = "Atrial flutter with variable AV condition"  
#             if RR_Rythm == "Regular" and P_wave == "False":
#                 self.result = "P wave is not present__AVNRT"    
#             if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate < ventricular_rate and RP_interval < PR_interval and RP_interval < 70:
#                 self.result = "AVNRT"
#             if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate < ventricular_rate and RP_interval > PR_interval:
#                 self.result = "Atypical AVNRT"          
#         if Heart_Rate > 100 and QRS_Duration > 120 :
           
#            if RR_Rythm == "Regular" and\
#               AV_relationship != 1 and ventricular_rate > Atrial_rate:
#                 self.result = "Ventricular_Tachycardia"
#            elif RR_Rythm == "Regular" and AV_relationship == 1 and R_peak_time >= 50:
#                 self.result = "Ventricular Tachycardia"
           
#            if RR_Rythm == "Regular" and AV_relationship != 1 and ventricular_rate < Atrial_rate :
#                 self.result = "Atrial flutter with variable AV condition"
#            elif  Heart_Rate > 100 and QRS_Duration > 120 and\
#                     RR_Rythm== "Regular" and  AV_relationship!= 1 and\
#                         ventricular_rate < Atrial_rate and\
#                         200 < Atrial_rate < 350 and QRS_Duration >= 120:
#                        self.result("Atrial flutter with aberrant conduction ,or ventricular premature complexes ")
#            elif RR_Rythm == "Irregular" and  Atrial_rate/ventricular_rate < 0.4 and\
#                 75 < ventricular_rate < 150:
#                      self.result = "Atrial flutter"      
#            elif RR_Rythm == "Irregular":
#                 self.result = "Premature Ventricular Complex "
#         if 50 <= Heart_Rate <= 100 and RR_Rythm == "Irregular" and P_wave == "False" and  QRS_Duration <= 120:
#             self.result = "Atrial fibrillation"
               
#         elif 100 <= Heart_Rate and RR_Rythm == "Irregular" and P_wave == "False":
#             self.result = "Atrial fibrillation with rapid ventricular response "
#         elif 50 > Heart_Rate and RR_Rythm == "Irregular" and P_wave == "False":
#             self.result = "Atrial fibrillation with slow ventricular response "
#         elif 50 >  Heart_Rate and RR_Rythm == "Irregular" and  P_wave == "False" and QRS_Duration>= 120 :
#             self.result(" Atrial fibrillation with slow ventricular response  with aberrant conduction ,or ventricular premature complexes")       
#         elif 50 <=  Heart_Rate < 100 and RR_Rythm == "Irregular" and\
#             P_wave == "False" and QRS_Duration>= 120:
#             self.result=(" Atrial fibrillation with aberrant conduction,or ventricular premature complexes") 
                     
#         else:
#             self.result += "\t" + ""
     
#         return self.result

# if __name__ == "__main__":
#     print("Hello from Arr")




import matplotlib.pyplot as plt
import numpy as np
import pywt
import pandas as pd
from utils import function as func
# import SNV


def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
    coeffs = pywt.wavedec(X, dwt_transform, level=dlevels)  # wavelet transform 'bior4.4'
    for ca in range(0, cutoff_low):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])

    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])
    Y = pywt.waverec(coeffs, dwt_transform)  # inverse wavelet transform
    return Y


   
class Arrhythmia_classification:
    def __init__(self, cleaned_signal,r_peak_index_val): 
        self.result = ""
        self.cleaned_signal = cleaned_signal
        self.r_peak_index_val=r_peak_index_val

    def classify_data(self, RHYTHM, lead):
        
        """
        Parameters:
            :param RHYTHM: result Data
            :param lead: name of lead
            :return: result
        """
        Heart_Rate = RHYTHM[lead]["Heart_Rate"]["V"]
        RR_Rythm = RHYTHM[lead]["RR_Rhythm"]["V"]
        QRS_Duration = RHYTHM[lead]["QRS_Duration"]["V"]
        PR_interval = RHYTHM[lead]["PR_Interval"]["V"]
        RP_interval = RHYTHM[lead]["RP_Interval"]["V"]
        Atrial_rate = RHYTHM[lead]["Atrial_rate"]["V"]
        ventricular_rate = RHYTHM[lead]["Ventricular_rate"]["V"]
        p_amp = RHYTHM[lead]["P_Amplitude"]["V"]
        q_amp = RHYTHM[lead]["Q_Amplitude"]["V"] 
        # PRinterval = RHYTHM[lead]["PR_Interval"]["V"]
        AV_relationship = RHYTHM[lead]["AV_relationship"]["V"]
        R_peak_time = RHYTHM[lead]["R_peak_time"]["V"] 
        RR_Deviation = RHYTHM[lead]["RR_Deviation"]["V"] 
        P_Duration = 49
        # VAT = RHYTHM[lead]["VAT"]["V"]
        VAT = 49
        RR_Interval = RHYTHM[lead]["RR_Interval"]["V"]
        if p_amp >= 0.1:
            P_wave = "True"
        else:
            P_wave = "False"
        if  q_amp >= -0.1:
                Q_wave = "True"
        else:
                Q_wave = "False"   
           
        # print(f"Rhythm: {RR_Rythm}, Heart Rate: {Heart_Rate}")

        if 60 <= Heart_Rate < 100 and RR_Rythm == "Regular" :
            self.result = "Normal sinus Rhythm"
        if Heart_Rate < 60  and RR_Rythm == "Regular":
          self.result = "Sinus Bradycardia"
        if Heart_Rate > 100  and RR_Rythm == "Regular":
          self.result = "Sinus Tachycardia"   
        if Heart_Rate <= 70 and  PR_interval<=140 and RR_Rythm == "Irregular":
            self.result = "Junctional Rhythm"
        if Heart_Rate > 70 and  PR_interval<=140 and RR_Rythm == "Irregular" :
            self.result = "Rapid Junctional Rhythm" 
       
        if  RR_Deviation > 0.2 * RR_Interval and RR_Interval>120 :
                self.result = "Sinus Arrhythmia"  
        if RR_Deviation> 0.4* RR_Interval and RR_Interval>120  and  RR_Rythm == "Irregular": 
                self.result = "Marked Sinus Arrhythmia"  
        if 50<=Heart_Rate ==100 and  PR_interval > 140 and p_amp <0.25:
                self.result = "Atrial Rhythm " 
         
        # if Heart_Rate > 70 and PR_interval > 140:
        #      self.result = "Rapid Atrial Rhythm "      
        if Heart_Rate >= 100 and QRS_Duration <= 120:  
            if RR_Rythm == "Regular" and P_wave == "True" :
                self.result = "Sinus Tachycardia"   
            if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate > ventricular_rate and 200 < Atrial_rate < 350:
                self.result = "Atrial flutter" 
            if  RR_Rythm == "Irregular" and 75 < ventricular_rate < 150:
                self.result = "Atrial flutter with variable AV condition"  
            if RR_Rythm == "Regular" and P_wave == "False":
                self.result = "AVNRT"    
            if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate < ventricular_rate and RP_interval < PR_interval and RP_interval < 70:
                self.result = "AVNRT"
            if RR_Rythm == "Regular" and P_wave == "True" and Atrial_rate < ventricular_rate and RP_interval > PR_interval:
                self.result = "Atypical AVNRT"          
        if Heart_Rate >= 100 and QRS_Duration > 120 :
           
           if RR_Rythm == "Regular" and\
              AV_relationship != 1 and ventricular_rate > Atrial_rate:
                self.result = "Ventricular_Tachycardia"
           elif RR_Rythm == "Regular" and AV_relationship == 1 and R_peak_time >= 50:
                self.result = "Ventricular Tachycardia"
           
        if RR_Rythm == "Regular" and AV_relationship != 1 and ventricular_rate < Atrial_rate :
                self.result = "Atrial flutter with variable AV condition"
        elif  Heart_Rate > 100 and QRS_Duration > 120 and\
                    RR_Rythm== "Regular" and  AV_relationship!= 1 and\
                        ventricular_rate < Atrial_rate and\
                        200 < Atrial_rate < 350 and QRS_Duration >= 120:
                       self.result("Atrial flutter with aberrant conduction ,or ventricular premature complexes ")
        if RR_Rythm == "Regular" and QRS_Duration > 120 and Atrial_rate/ventricular_rate <1.1 and Atrial_rate/ventricular_rate!=1 and\
            75 < Heart_Rate < 150:
            self.result = "Atrial flutter"      
        #    elif RR_Rythm == "Irregular":
                # self.result = "Premature Ventricular Complex "
        if 50 <= Heart_Rate <= 100 and RR_Rythm == "Irregular" and (P_wave == "False" or RR_Deviation> 0.125* RR_Interval) :
                self.result = "Atrial fibrillation"
               
        elif 100 <= Heart_Rate and RR_Rythm == "Irregular" and (P_wave == "False" or RR_Deviation> 0.125* RR_Interval):
            self.result = "Atrial fibrillation with rapid ventricular response "
        elif 50 > Heart_Rate and RR_Rythm == "Irregular" and (P_wave == "False" or RR_Deviation> 0.125* RR_Interval):
            self.result = "Atrial fibrillation with slow ventricular response "
        elif 50 >  Heart_Rate and RR_Rythm == "Irregular" and  P_wave == "False" and QRS_Duration>= 120 :
            self.result(" Atrial fibrillation with slow ventricular response  with aberrant conduction ,or ventricular premature complexes")       
        elif 50 <=  Heart_Rate < 100 and RR_Rythm == "Irregular" and\
            P_wave == "False" and QRS_Duration>= 120:
            self.result=(" Atrial fibrillation with aberrant conduction,or ventricular premature complexes") 
       
        if self.result == "":
            self.result="Undetermine"

        else:
            self.result += "\t" + ""
     
        return self.result

if __name__ == "__main__":
    print("Hello from Arr")