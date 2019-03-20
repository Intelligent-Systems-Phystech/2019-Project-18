import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_res(y_test, y_pred, outfile='../doc/pictures/pls_results.pdf'):
    plt.rc('font', size=30)
    axes = ['$x$', '$y$', '$z$']
    plt.figure(figsize=(30, 30))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.ylabel(axes[i], rotation=0, labelpad=20)
        plt.xlabel('$t$', labelpad=20)
        plt.plot(y_test[0:1000,i], label='наблюдаемое движение', linewidth=3.0)
        plt.plot(y_pred[0:1000,i], label='предсказание', linewidth=3.0)
        plt.grid(True)
        plt.legend(loc='upper right')
    if outfile == None:
        plt.show()
    else:
        plt.savefig(outfile)