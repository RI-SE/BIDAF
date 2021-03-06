import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.interactive(True)
mpl.rcParams['toolbar']='None'

class WindowMgr():
    def __init__(self, name, width, height, numx, numy, marg = 0, dir='horizontal'):
        self.maxind = (numx,numy)
        self.dir = dir
        self.dxm = marg/width
        self.dym = marg/height
        self.dx = (1.0-self.dxm)/self.maxind[0]
        self.dy = (1.0-self.dym)/self.maxind[1]
        self.nextind = (0,0)
        self.targetdict = {}
        self.lastkeytarget = None
        self.lastbuttontarget = None
        self.globalkeydict = {}
        self.fig = plt.figure(name)
        self.fig.set_size_inches((width/self.fig.dpi, height/self.fig.dpi))
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.fig.canvas.mpl_connect('key_release_event', self.key_release_callback)
        self.fig.canvas.mpl_connect('scroll_event', self.scroll_callback)
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('motion_notify_event', self.button_motion_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)

    def get_figure(self):
        return self.fig
        
    def get_next_rect(self):
        (nx, ny) = self.nextind
        if nx<0 or ny<0:
            return False
        rect = (nx*self.dx+self.dxm, 1.0-(ny+1)*self.dy, self.dx-self.dxm, self.dy-self.dym)
        if self.dir=='vertical':
            ny += 1
            if ny>=self.maxind[1]:
                ny = 0
                nx += 1
                if nx >= self.maxind[0]:
                    nx = -1
        else:
            nx += 1
            if nx>=self.maxind[0]:
                nx = 0
                ny += 1
                if ny >= self.maxind[1]:
                    ny = -1
        self.nextind = (nx, ny)
        return rect

    def register_target(self, rect, target):
        self.targetdict[rect] = target

    def clear_targets(self):
        self.targetdict = {}
        self.nextind = (0,0)

    def get_callback_target(self, event):
        pos = self.fig.transFigure.inverted().transform((event.x, event.y))
        for rect in self.targetdict:
            if pos[0]>=rect[0] and pos[0]<rect[0]+rect[2] and pos[1]>=rect[1] and pos[1]<rect[1]+rect[3]:
                return self.targetdict[rect]
        return None

    def install_key_action(self, key, func):
        self.globalkeydict[key] = func

    def key_press_callback(self, event):
        #print("Key pressed: " + event.key)
        if event.key in self.globalkeydict:
            self.globalkeydict[event.key]()
        else:
            self.lastkeytarget = self.get_callback_target(event)
            if self.lastkeytarget is not None and 'key_press_event' in dir(self.lastkeytarget):
                self.lastkeytarget.key_press_event(event)

    def key_release_callback(self, event):
        # The release goes to the same target as the press
        if self.lastkeytarget is not None and 'key_release_event' in dir(self.lastkeytarget):
            self.lastkeytarget.key_release_event(event)

    def scroll_callback(self, event):
        # Scrolls are special - no release
        target = self.get_callback_target(event)
        if target is not None and 'scroll_event' in dir(target):
            target.scroll_event(event)

    def button_press_callback(self, event):
        #print("Button pressed: " + str(event.button))
        self.lastbuttontarget = self.get_callback_target(event)
        if self.lastbuttontarget is not None and 'button_press_event' in dir(self.lastbuttontarget):
            self.lastbuttontarget.button_press_event(event)

    def button_motion_callback(self, event):
        # The motion goes to the same target as the press. Only motion events while pressed 
        if self.lastbuttontarget is not None and 'motion_notify_event' in dir(self.lastbuttontarget):
            self.lastbuttontarget.motion_notify_event(event)

    def button_release_callback(self, event):
        # The release goes to the same target as the press
        if self.lastbuttontarget is not None and 'button_release_event' in dir(self.lastbuttontarget):
            self.lastbuttontarget.button_release_event(event)
        self.lastbuttontarget = None
