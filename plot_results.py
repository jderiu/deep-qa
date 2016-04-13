import cPickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#roughness: sd(diff(x))/abs(mean(diff(x)))
def main():
    test_2016n = 'Test 2016'
    test_2015n = 'Test 2015'
    test_2014n = 'Test 2014'
    test_2013n = 'Test 2013'
    test_2014ljn = 'Test 2014 LiveJournal'
    test_2014srcn = 'Test 2014 Sarcasm'
    test_2013_smsn = 'Test 2013 SMS'
    train_fulln = 'Training Score'
    pdist_n = 'parameter distance'

    lab =['Test 2016','Training Score']
    names = [test_2016n,train_fulln]
    colors = ['#FF00FF','#001EFF','#00E6FF']

    eps_values = [1,2,3,4,5,6,7,8,9,10]
    rho_values = [5,25,50,75,95]

    stats = []
    for eps in eps_values:
        for rho in rho_values:
            ty = 'L3Aadadelta0eps{}rho{}'.format(eps,rho)
            try:
                resmap = cPickle.load(open('sup_res/AdadeltaTest/supervised_results_{}.p'.format(ty),'rb'))
            except:
                continue
            pp = PdfPages('adadelta_plots/{}.pdf'.format(ty))
            cnt = 0

            for k in resmap.keys():
                res_k = resmap[k]
                if not res_k:
                    continue
                mxidx = res_k.index(max(res_k))
                print 'Name:',k,'Epoch',(mxidx + 1)/5.0,'Val:',max(res_k)

                if k in names:
                    x = np.linspace(start=0.2,stop=len(res_k)/5.0,num=len(res_k))
                    y = np.array(res_k)
                    mx = max(res_k)
                    mxidx = res_k.index(mx)
                    #plt.plot([(mxidx + 1)/5.0],[mx],'o',color=colors[cnt],markersize=7,zorder=2)
                    plt.plot(x,y,'-',label=lab[cnt],lw=1.5,zorder=0,color=colors[cnt])
                    cnt += 1
                if k == pdist_n:
                    x = np.linspace(start=0.2,stop=len(res_k)/200,num=len(res_k))
                    y = ((100)/(np.max(res_k) - np.min(res_k))) * (np.array(res_k) - np.max(res_k)) + 100
                    plt.plot(x,y,'-',label=lab[cnt],lw=1.5,zorder=0,color=colors[cnt])
                    cnt += 1
                if k == test_2016n:
                    mx_2016 = max(res_k)

            x_train = np.asarray(resmap[train_fulln])
            x_test = np.asarray(resmap[test_2016n])

            print np.corrcoef(x_train,x_test)
            print 'Mean:',np.mean(x_train), 'Std:',np.std(x_train)
            print 'Mean:',np.mean(x_test), 'Std:',np.std(x_test)

            smoothness = 1.0/(np.std(np.diff(x_test))/(np.abs(np.mean(np.diff(x_test)))*len(x_test)))
            stats.append((eps,rho,mx_2016,smoothness))

            #plt.title('F1 score over time for {}'.format(ty))
            plt.xlabel('Number of Epochs')
            plt.ylabel('F1 score')
            plt.legend(loc=4, borderaxespad=0.2)
            pp.savefig()
            pp.close()
            plt.close()

    for t in stats:
        print t

    eps_x = map(lambda x: x[0],stats)
    rho_x = map(lambda x: x[1]/100.0,stats)
    max_z = map(lambda x: x[2],stats)

    print eps_x
    E,R = np.meshgrid(eps_x,rho_x)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(eps_x,rho_x,max_z,cmap=cm.jet, linewidth=0.2)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    print np.corrcoef(eps_x,max_z)
    print np.corrcoef(rho_x,max_z)

    plt.plot(eps_x,max_z)
    plt.show()

    return
    name = 'sup_res/Round1/supervised_results_L{}T{}Wcustom{}.p'

    layer_arch = [(1,'','L1'),(2,'','L2B'),(2,'A2','L2A'),(3,'','L3'),(3,'A2','L3A')]
    layer_arch = [(1,'','L1'),(2,'A2','L2A'),(3,'A2','L3A')]
    layer_arch = [(3,'A2','L3A')]
    ntweets = [0,1,2,4,10,50,90]
    ntweets = [0,10,85]#,1,2,4,10,50,85]
    colors = ['b','g','r','c','m','y','k']
    #colors = ['#009900','#000099','#990000']
    colors = ['#00E6FF','#FF00FF','#001EFF']
    #colors = ['#999999','#4d4d4d','#000000']
    linestyles = [':','--','-']

    jumpidx_layerarch = {}
    jumpnt_layarch = {}

    mxidx_layerarch = {}
    mx_layarch = {}
    for layer,arch,lab in layer_arch:
        layer_arch = lab

        jumpidx_layerarch[layer_arch] = []
        jumpnt_layarch[layer_arch] = []

        mxidx_layerarch[layer_arch] =  []
        mx_layarch[layer_arch] = []

        for nt,c in zip(ntweets,colors):
            resmap = cPickle.load(open(name.format(layer,nt,arch),'rb'))
            res_k = resmap[test_2016n]
            st = np.linspace(start=0,stop=245,num=len(res_k)/5.0).astype('int')
            end = np.linspace(start=4,stop=249,num=len(res_k)/5.0).astype('int')
            smooth_res = []
            for s,e in zip(st,end):
                m = np.median(res_k[s:e])
                smooth_res.append(m)

            if nt == 85:
                nt = 90

            oldval1 = np.inf
            oldval2 = np.inf
            jumpidx = 0
            jumpval = 0

            x = np.linspace(start=0.2,stop=len(res_k)/5,num=len(res_k))
            y = np.array(res_k)
            for idx, val in enumerate(y):
                if (val - oldval1 > 10 or val - oldval2 > 25) and val > 50:
                    jumpidx = idx
                    jumpval = val
                    jumpidx_layerarch[layer_arch].append(idx)
                    jumpnt_layarch[layer_arch].append(nt)
                    break
                oldval1 = val
                oldval2 = oldval1

            if not nt == 0:
                lb = '{}M distant tweets '.format(nt)
            else:
                lb = 'Without distant supervised training'

            plt.plot(x,y,'-',label=lb,color=c,lw=1.5,zorder=0)
            mx = max(res_k)
            mxidx = res_k.index(mx)
            mx_layarch[layer_arch].append(nt)
            mxidx_layerarch[layer_arch].append(mxidx)

            plt.plot([(mxidx + 1)/5.0],[mx],'o',color=c,markersize=7,zorder=2)
            #plt.plot([(jumpidx + 1)/5.0],[jumpval],'x',color=c,zorder=1,markersize=7)
            #plt.axvline((jumpidx + 1)/5.0,color=c,lw=3.0,zorder=1)
            print 'Architecture L{}T{},Number of tweets:'.format(layer,arch),nt,'Index:',mxidx,'Epoch',(mxidx + 1)/5.0,'Value:',mx

        plt.title('F1 score over time')
        plt.xlabel('Number of Epochs')
        plt.ylabel('F1 score')
        plt.legend(loc=4, borderaxespad=0.2)
        plt.show()

    colors = ['#000000','#333333','#4d4d4d']
    stype = ['o','s','^','<','>','d']
    for k,c,t in zip(jumpidx_layerarch.keys(),colors,stype):
        mx = mx_layarch[k]
        mxidx = mxidx_layerarch[k]

        plt.plot(mx,map(lambda x: (x + 1)/5.0,mxidx),'{}-'.format(t),lw=1.6,zorder=0,markersize=8,label=k,color=c)
        plt.xlabel('Number of tweets')
        plt.ylabel('Jump Epoch')
        plt.legend(loc=1, borderaxespad=0.2)
    plt.show()

if __name__ == '__main__':
    main()