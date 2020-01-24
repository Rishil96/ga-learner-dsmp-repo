# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#Code starts here
data = pd.read_csv(path)
loan_status = data["Loan_Status"].value_counts()
loan_status.plot.bar()


# --------------
#Code starts here
property_and_loan = data.groupby(["Property_Area","Loan_Status"]).size().unstack()
property_and_loan.plot.bar(stacked=False)
plt.xlabel("Property Area")
plt.ylabel("Loan Status")
plt.xticks(rotation=45)


# --------------
#Code starts here
education_and_loan = data.groupby(["Education","Loan_Status"]).size().unstack()
education_and_loan.plot.bar(stacked=True)
plt.xlabel("Education Status")
plt.ylabel("Loan Status")
plt.xticks(rotation=45)


# --------------
#Code starts here
graduate = data[data["Education"]=='Graduate']
not_graduate = data[data["Education"]=='Not Graduate']
graduate["LoanAmount"].plot(kind='density',label="Graduate")
not_graduate["LoanAmount"].plot(kind='density',label="Not Graduate")




#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
data['TotalIncome'] = data["ApplicantIncome"] + data["CoapplicantIncome"]
fig ,(ax_1,ax_2,ax_3) = plt.subplots(nrows=3,ncols=1,figsize=(15,10))
data.plot.scatter(x='ApplicantIncome',y='LoanAmount',ax=ax_1,title="Applicant Income")
data.plot.scatter(x='CoapplicantIncome',y='LoanAmount',ax=ax_2,title="Coapplicant Income")
data.plot.scatter(x='TotalIncome',y='LoanAmount',ax=ax_3,title="Total Income")



