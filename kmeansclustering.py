import pandas as pd
from datetime import datetime as dt
from calendar import isleap as is_leap_year
from sklearn.preprocessing import MinMaxScaler
'''
Author : YongBaek Cho
'''

def euclidean_distance(v1, v2):
    """ Computes the Euclidean distance between two vectors.

    Args:
        v1, v2 (Series): vectors of the same length with the same attributes.

    Returns:
        distance (float): the Euclidean distance.
    
    """
    distance = (sum([(v1[i]-v2[i])**2 for i in range(len(v1))]))**(1/2)
    return distance

def make_frame():
    """ Creates a DataFrame from csv file 'TIA_1987_2016.csv', indexes it with
        dates beginning from 1987-01-01.

    Returns:
        frame (DataFrame): DataFrame object from csv file.
    
    """
    frame = pd.read_csv('TIA_1987_2016.csv')
    frame.index = pd.date_range(dt(1987, 1, 1), periods=len(frame))
    return frame

def clean_dewpoint(frame):
    """ Takes the frame created by make_frame() function and replaces values
        '-999' from the "Dewpt" column with average value of the dewpoints on
        same days during 29 years.

    Args:
        frame (DataFrame): frame created by make_frame() function.
    
    """
    clean_dates = [i for i in frame[frame['Dewpt']==-999].index]
    for date in clean_dates:
        dr = pd.date_range(start='1987-'+str(date.month)+'-'+str(date.day),
                           periods=30,
                           freq=pd.DateOffset(years=1))
        mean = frame['Dewpt'][dr.drop(date)].mean()
        frame['Dewpt'][date] = mean

def day_of_year(date_time):
    """ This function takes a datetime object. Return the day of the year
        it represents as a number between 1 and 365. If it is a leap year,
        return the day of the year as though it were not, unless
        the date is February 29, in which case, return 366.

        Args:
            date_time (datetime): datetime object.

        Returns:
            day (int): number of day in the year (366 for February 29).

    """
    day = date_time.timetuple().tm_yday
    if not is_leap_year(date_time.year):
        return day
    else:
        if day==60:
            return 366
        elif day<60:
            return day
        else:
            return day-1

def climatology(frame):
    """ This function takes a datetime object. Return the day of the year
        it represents as a number between 1 and 365. If it is a leap year,
        return the day of the year as though it were not, unless
        the date is February 29, in which case, return 366.

        Args:
            frame (DataFrame): frame created by make_frame() function.

        Returns:
            cframe (DataFrame): frame of average values for each parameter
                for each day of the year.

    """
    cframe = frame.groupby(day_of_year).mean()[:365]
    return cframe
    
def scale(frame):
    """ Rescales every value in frame in range between 0 and 1.

        Args:
            frame (DataFrame): frame created by climatology() function.

    """
    scaler = MinMaxScaler(copy=False)
    scaler.fit_transform(frame)

def get_initial_centroids(frame, k):
    """ Picks initial centroids from a frame.

        Args:
            frame (DataFrame): frame created by climatology() function.
            k (int): number of centroids.

        Returns:
            centroids (DataFrame): frame of centroids.
        
    """
    l = [list(frame.loc[i*int(len(frame)/k)+1]) for i in range(k)]
    centroids = pd.DataFrame(l, index=range(k),
                        columns=["Dewpt", "AWS", "Pcpn", "MaxT", "MinT"])
    return centroids

def classify(centroids, vector):
    """ Finds which centroid is the closest to the vector.

        Args:
            centroids (DataFrame): frame of centroids.
            vector (Series): row from the climo frame.

        Returns:
            label (int): label of the cluster to which the vector is assigned.
        
    """
    l = [pd.Series(centroids.loc[i]) for i in range(len(centroids))]
    label = min(l, key=lambda x: euclidean_distance(list(x), vector)).name
    return label

def get_labels(frame, centroids):
    """ Creates the list of labels for each index in the frame.

        Args:
            frame (DataFrame): frame created by climatology() function.
            centroids (DataFrame): frame of centroids.

        Returns:
            labels (Series): list of labels that maps the indices of the frame
                to the labels of the clusters.
        
    """
    l = [classify(centroids, frame.loc[i]) for i in frame.index]
    labels = pd.Series(l, index=frame.index)
    return labels

def update_centroids(frame, centroids, labels):
    """ Updates centroids by placing them in the center of their current cluster.

        Args:
            frame (DataFrame): frame created by climatology() function.
            centroids (DataFrame): frame of centroids.
            labels (Series): list of labels that maps the indices of the frame
                to the labels of the clusters.

    """
    centroids[:] = 0.0
    for index in frame.index:
        centroids.loc[labels.loc[index]] += frame.loc[index]

    series = labels.value_counts()

    for i in range(len(centroids)):
        centroids.loc[i] /= series[i]

def k_means(frame, k):
    """ Performs k-means algorithm on given frame and separates data into
        k clusters.

        Args:
            frame (DataFrame): frame created by climatology() function.
            k (int): number of centroids.

        Returns:
            centroids (DataFrame): frame of centroids.
            labels (Series): list of labels that maps the indices of the frame
                to the labels of the clusters.
        
    """
    centroids = get_initial_centroids(frame, k)
    labels = get_labels(frame, centroids)
    while True:
        update_centroids(frame, centroids, labels)
        labels_new = get_labels(frame, centroids)
        if (abs(labels.values - labels_new.values) < 0.001).all():
            labels = labels_new
            return centroids, labels
        labels = labels_new
