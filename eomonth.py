from pandas.tseries.offsets import MonthEnd

import datetime
from dateutil.relativedelta import relativedelta
def eomonth(date, offset):

    month = (date.month + offset%12)%12

    addYear = offset // 12 + 1 if month  == 0 else offset // 12

    dt = date.replace(year=date.year+addYear).replace(month=month + 1).replace(day=1) + datetime.timedelta(days=-1)

    dt = (date + relativedelta(months=offset+1)).replace(day=1) + datetime.timedelta(days=-1)

    return dt


def date_diff(interval, p_datefrom , p_dateto ):
    if interval =='q':
        return (((p_dateto.year * 12) + p_dateto.month) - ((p_datefrom.year * 12) + p_datefrom.month))//3
    if interval=='m':
        return ((p_dateto.year * 12) + p_dateto.month) - (  (p_datefrom.year* 12) + p_datefrom.month )
    return None

for a in range(-24, 25):
        print(eomonth(datetime.date(2019, 6, 22),0), eomonth(datetime.date(2019, 6, 22), a))
        print("date diff: ", date_diff('q', datetime.date(2019, 6, 22), eomonth(datetime.date(2019, 6, 22), a)))
