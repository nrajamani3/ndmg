"""A fun little tool to look at the largest connected components
overlaid on top of the faMaps"""

from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons

import numpy as np
import argparse

import lcc
import fa
    
def show_figure(bfn):
    faDir = "/mnt/braingraph1data/projects/will/mar12data/fa/"
    ccDir = '/data/biggraphs/connectedcomp/'
    
    fafn =  faDir+bfn+"_fa"
    faXML = fa.FAXML(fafn+'.xml')
    faData = fa.FAData(fafn+'.raw',faXML.getShape())
    vcc = np.load(ccDir+bfn+'_concomp.npy')
    cc3d = lcc.get_3d_cc(vcc,faXML.getShape())
    
    sl = 100
    xyz = 'xy'
    
    
    fig = figure();
    
    ax = subplot(111)
    subplots_adjust(left=0.25, bottom=0.25)
    
    
    def draw_brain(sl,xyz):
        ax = subplot(111)
        ax.clear()
        lcc.show_overlay(faData.data,cc3d,ncc=15,s=sl,xyz=xyz)
        draw()
        
    draw_brain(sl,xyz)
    
    
    
    axcolor = 'lightgoldenrodyellow'
    axslice = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    
    sslice = Slider(axslice, 'Slice', 0, cc3d.shape[0]-1, valinit=sl)
    
    def update_slice(val):
        global xyz
        global sl
        sl = val
        
        draw_brain(sl,xyz)
        
    sslice.on_changed(update_slice)
    
    
    rax = axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
    radio = RadioButtons(rax, ('xy', 'xz', 'yz'), active=0)
    def xyzproj(newxyz):
        global xyz
        global sl
        global sslice
        
        
        xyz = newxyz
        dimDict = {'xy':2,'xz':1,'yz':0}
        sl = int(cc3d.shape[dimDict[xyz]]/2)
        
        sslice.valmax = cc3d.shape[dimDict[xyz]]
        sslice.set_val(sl)
        draw()
        
        draw_brain(sl,xyz)
        
    radio.on_clicked(xyzproj)
    
    show()



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Show the lcc overlayed on the fa map')
    parser.add_argument('brainFn', action="store")
    result = parser.parse_args()
    
    bfn = result.brainFn
    show_figure(bfn)