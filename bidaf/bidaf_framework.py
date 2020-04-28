import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import windowmgr as wm


def findbestlayout(num):
    tmp = int(np.ceil(np.sqrt(num)*2))
    if tmp%2==1:
        return (int((tmp+1)/2), int((tmp-1)/2))
    else:
        return (int(tmp/2), int(tmp/2))

class BidafFramework():
    def __init__(self, datasource, models, visualizers, settings):
        (h, v) = findbestlayout(len(visualizers))
        self.mgr = wm.WindowMgr("BIDAF", 1400, 1000, h, v, 50, 'vertical')
        self.fig = self.mgr.get_figure()
        self.data_repo = datasource
        self.data_source = datasource
        self.models = []
        self.versions = {}
        self.visualizers = []
        self.callbacks = {}
        self.timer = self.fig.canvas.new_timer(interval = 30)
        self.timer.single_shot = True
        self.timer.add_callback(self.inner_loop)
        self.stopped = True
        self.mgr.install_key_action("enter", self.toggle_run)

        for modcl in models:
            mod = modcl()
            mod.set_params(settings)
            mdict = mod.extract_model() 
            self.versions[mdict['type']] = mdict['version']
            self.models.append(mod)

        hack=0
        d=0.25
        for viscl in visualizers:
            rect = self.mgr.get_next_rect()
            if hack==0:
                rect = (rect[0], rect[1]+d, rect[2], rect[3]-d)
                hack+=1
            elif hack==1:
                rect = (rect[0], rect[1], rect[2], rect[3]+d)
                hack+=1
            vis = viscl(self.fig, rect, self.data_repo, self.callbacks)
            vis.set_params(settings)
            self.mgr.register_target(rect, vis)
            self.visualizers.append(vis)
            

    def handle_message(self, dict_msg):
        # 0 - init original features names
        # self.data_repo.set_original_feature_names([name for name in dict_msg.keys() if name not in ['time','entity']])
        # 1 - send data to models, collecting the resulting features
        resultlist = [m.handle_data(dict_msg) for m in self.models]
        # 2 - add all results to dictmsg and add it to repo
        for res in resultlist:
            if res is not None:
                dict_msg.update(res)
        # self.data_repo.add(dict_msg)
        # 3 - update old messages and/or generate new model
        newmodels = []
        features_changed = False
        for m in self.models:
            # get new models
            if m.model_version() > self.versions[m.model_type()]:
                newmodels.append(m.extract_model())
                self.versions[m.model_type()] = m.model_version()
            # update new features
            if m.features_changed():
                self.data_repo.update(m.update_features())
                # We need to redraw visualizations
                features_changed = True
        # 4 - Visualize current message
        for vis in self.visualizers:
            vis.handle_data(dict_msg)
        # 5 - visualize models
        for nmod in newmodels:
            for vis in self.visualizers:
                vis.redraw_model(nmod)
        # 6 - visualize updated features (only if a module changed old features)
        features_changed = True ## Checking interaction
        if features_changed:
            for vis in self.visualizers:
                vis.redraw_features()
        # 7 - Now we draw for every message. And clean up.
        # self.data_repo.cleanup()
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
        # Done processing modules

    def handle_batch_data(self, dictdata):
        # 0 - init original features names
        # if len(dictdata) > 0:
        #     self.data_repo.set_original_feature_names([name for name in dictdata[0].keys() if name not in ['time','entity']])
        # 1 - send data to models, collecting the resulting features
        resultlist = [m.handle_batch_data(dictdata) for m in self.models]
        # 2 - add all results to dictmsg and add it to repo
        # for dict_msg in dictdata:
        #     self.data_repo.add(dict_msg)
        for res in resultlist:
            if res is not None:
                self.data_repo.update(res)
        # 3 - generate new model
        newmodels = []
        for m in self.models:
            # get new models
            if m.model_version() > self.versions[m.model_type()]:
                newmodels.append(m.extract_model())
                self.versions[m.model_type()] = m.model_version()
        # 4 - Don't visualize current messages now, redraw_features below
        # 5 - visualize models
        for nmod in newmodels:
            for vis in self.visualizers:
                vis.redraw_model(nmod)
        # 6 - visualize updated features (only if a module changed old features)
        for vis in self.visualizers:
            vis.redraw_features()
        # 7 - Now we draw for every message. And clean up. Batch is unlimited.
        #self.data_repo.cleanup()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Done processing modules

    def inner_loop(self):
        if not self.stopped and self.data_source.available():
            dict_msg = self.data_source.next()
            self.handle_message(dict_msg)
            plt.draw()
        if not self.stopped: # might have been stopped meanwhile
            self.timer.start()

    def toggle_run(self):
        if self.stopped:
            self.start()
        else:
            self.stop()

    def start(self):
        self.stopped = False
        self.timer.start()

    def stop(self):
        self.stopped = True
        self.timer.stop()

    def run_batch(self):
        data = self.data_source.all()
        self.handle_batch_data(data)
