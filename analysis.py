import pandas as pd
import matplotlib.pyplot as plt

def analysis(data):
    plt.close('all')
    
    column = 'longitude'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
    
    column = 'latitude'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
    
    column = 'housing_median_age'
    plot = data[column].plot.hist(bins=50)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
    
    column = 'total_rooms'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
    
    column = 'total_bedrooms'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
              
    column = 'population'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
    
    column = 'households'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
              
    column = 'median_income'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
              
    column = 'ocean_proximity'
    plot = data[column].value_counts().plot.bar()
    plt.xlabel(column)
    plt.xticks(rotation=15, ha="right")
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')

    column = 'median_house_value'
    plot = data[column].plot.hist(bins=100)
    plt.xlabel(column)
    fig = plot.get_figure()
    fig.savefig('graphs/' + column + ".png")
    plt.close('all')
    
    print("plotted")

def main():
    data = pd.read_csv("housing.csv") 
    analysis(data)


if __name__ == "__main__":
    main()