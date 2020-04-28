import tkinter
from tkinter import ttk, filedialog
from tkinter import * #Tk, filedialog, StringVar, W, Listbox
from tkinter.ttk import *

import pandas as pd

from tkinter.font import nametofont
from dateutil import parser
from datetime import datetime

from gaussfilter import *

class FileDialog:
    def __init__(self, name, horizon):
        self.root = Tk()
        self.root.title(name)
        self.horizon = horizon
        self.file_path = ''
        self.file_label = None
        self.sep = StringVar()
        self.start = False
        self.file_loaded = False
        self.df = None
        self.data_dict = None
        self.data_index = 0
        self.lookupkey = {}

        # We need preprocessing to implement modes and scales
        #self.modes = None
        #self.scales = None

        self.index = None
        self.entity = None
        self.time = None
        self.raw_features = None
        self.features = None
        self.features_all = None # entity + time + features
        self.scale = None
        self.slope = None
        self.variance = None

        self.entity_selector = None
        self.time_selector = None
        self.scale_lbl = None
        self.scale_ui = None
        self.scale_var = DoubleVar()
        self.stride_lbl = None
        self.stride_ui = None
        self.stride_var = DoubleVar()
        self.extra_lbl = None
        self.slope_ui = None
        self.slope_var = BooleanVar()
        self.variance_ui = None
        self.variance_var = BooleanVar()
        self.columns = []
        self.col_selector = None
        self.col_selector_var = StringVar()
        self.start_btn = None

        self.filter = None
        self.delivered = False

        self.show()
        self.root.mainloop()
        self.root.destroy()
        self.root = None

    def show(self):

        ######## csv separator
        self.sep.set(',')  # initializing the choice
        # seps format is (<display text>, <char>)
        seps = [(',', ','), (';', ';'), ('tab', '\t')]

        Label(self.root, text="File separator:").grid(row=0, sticky=E)

        for i, s in enumerate(seps):
            Radiobutton(self.root, text=s[0], variable=self.sep, value=s[1]).grid(row=0, column=i+1, sticky=W)

        ######## file selector
        Label(self.root, text="Open File:").grid(row=1, sticky=E)
        Button(self.root, text='Select file', command=self.file_selector).grid(row=1, column=1, columnspan=2, sticky=W)
        self.file_label = Label(self.root, text="")
        self.file_label.grid(row=2, column=1, columnspan=3, sticky=W)

        ######## entity selector
        Label(self.root, text="Entity:").grid(row=3, sticky=E)
        self.entity_selector = Combobox(self.root, state="readonly", values='')
        self.entity_selector.grid(row=3, column=1, columnspan=3, sticky=W+E)
        self.entity_selector.bind("<<ComboboxSelected>>", self.option_selected)

        ######## time col selector
        Label(self.root, text="Time:").grid(row=4, sticky=E)
        self.time_selector = Combobox(self.root, state="readonly", values='')
        self.time_selector.grid(row=4, column=1, columnspan=3, sticky=W+E)
        self.time_selector.bind("<<ComboboxSelected>>", self.option_selected)


        ######## time properties
        self.scale_lbl = Label(self.root, text="Time Scale:", state=DISABLED)
        self.scale_lbl.grid(row=5, sticky=E)
        self.scale_var.set(10.0)
        self.scale_ui = Entry(self.root, textvariable=self.scale_var, width=6, state=DISABLED)
        self.scale_ui.grid(row=5, column=1, sticky=W)

        self.stride_lbl = Label(self.root, text="Time Stride:", state=DISABLED)
        self.stride_lbl.grid(row=6, sticky=E)
        self.stride_var.set(10.0)
        self.stride_ui = Entry(self.root, textvariable=self.stride_var, width=6, state=DISABLED)
        self.stride_ui.grid(row=6, column=1, sticky=W)

        self.extra_lbl = Label(self.root, text="Expand Features:", state=DISABLED)
        self.extra_lbl.grid(row=7, sticky=E)
        self.slope_var.set(False)
        self.slope_ui = Checkbutton(self.root, text='Slope', variable=self.slope_var, state=DISABLED)
        self.slope_ui.grid(row=7, column=1, sticky=W)
        self.variance_var.set(False)
        self.variance_ui = Checkbutton(self.root, text='Variance', variable=self.variance_var, state=DISABLED)
        self.variance_ui.grid(row=7, column=2, sticky=W)

        # Label(self.root, text="Variance:", state=DISABLED).grid(row=8, sticky=E)




        ######## other col selector
        Label(self.root, text="Features:").grid(row=9, sticky=E)
        self.col_selector = Listbox(self.root, selectmode='multiple', exportselection=0, listvariable=self.col_selector_var)
        self.col_selector.grid(row=9, column=1, columnspan=3, rowspan=6, sticky=W+E)


        ######## start & exit
        self.start_btn = Button(self.root, text='Start', state=DISABLED, command=self.start_exec)
        self.start_btn.grid(row=15, column=1, sticky=E)
        Button(self.root, text='Quit', command=self.root.quit).grid(row=15, column=2, sticky=W)


        ######## Padding to all
        for child in self.root.winfo_children():
            child.grid_configure(padx=10, pady=10)




    def enable_time_properties(self, enabled):
        state = NORMAL if enabled else DISABLED

        for item in [self.scale_lbl, self.scale_ui, self.stride_lbl, self.stride_ui, self.extra_lbl, self.slope_ui, self.variance_ui]:
            item['state'] = state

    def file_selector(self):
        self.file_path = filedialog.askopenfilename()

        try:
            self.df = pd.read_csv(self.file_path, sep=self.sep.get(), infer_datetime_format=True, keep_date_col=True)
            self.file_label['text'] = self.file_path
            self.file_loaded = True
            self.start_btn['state'] = NORMAL
        except:
            self.file_label['text'] = "Error loading file!"
            self.file_path = None
            self.file_loaded = False
            self.start_btn['state'] = DISABLED
            return

        print(list(self.df.columns))
        self.columns = list(self.df.columns)

        ######## Fill selectors
        self.time_selector['values'] = [''] + self.columns
        self.entity_selector['values'] = [''] + self.columns

        self.col_selector_var.set(self.columns)
        self.col_selector.select_set(0, END)

    def option_selected(self, event):
        e = self.entity_selector.current()
        t = self.time_selector.current()

        self.col_selector.select_set(0, END)
        if e > 0:
            self.col_selector.select_clear(e-1) # +1 because empty option in the list
        if t > 0:
            self.col_selector.select_clear(t-1)
            # enable time properties
            self.enable_time_properties(True)
        else:
            self.enable_time_properties(False)


    def __str__(self):
        return ("Time: {}, Entity: {}, Features: {}, File:{}".format(self.time, self.entity, self.features, self.file_path))

    def start_exec(self):
        # Column name as string or False if not selected
        self.entity = self.entity_selector.get() if self.entity_selector.current() > 0 else False
        self.time = self.time_selector.get() if self.time_selector.current() > 0 else False
        self.index = 'Index' if self.time is False else False

        # list of selected features names as string
        self.raw_features = [self.col_selector.get(i) for i in self.col_selector.curselection()]

        if self.time is not False:
            self.scale = self.scale_var.get()
            self.stride = self.stride_var.get()
            self.slope = self.slope_var.get()
            self.variance = self.variance_var.get()
        else:
            self.scale = False
            self.stride = False
            self.slope = False
            self.variance = False

        self.start = True
        self.root.quit()

    def deliver(self, dictvec):
        key = (dictvec[self.time], dictvec[self.entity]) if self.entity is not False else (dictvec[self.time],)
        self.lookupkey[key] = len(self.data_dict)
        self.data_dict.append(dictvec)
        self.delivered = True

    def prepare_data(self):
        if self.time is not False:
            if len(self.df) > 0:
                if type(self.df[self.time][0])==str:  # Assume date string
                    self.df[self.time] = self.df[self.time].apply(parser.parse).apply(datetime.timestamp)
                elif self.df[self.time][0] > 4e9:     # Most likely milliseconds
                    self.df[self.time] = self.df[self.time].apply(lambda x:x/1000)
            self.df = self.df.sort_values(by=[self.time])
            self.raw_data_dict = self.df.to_dict(orient='records')
            self.raw_data_index = 0
            self.filter = GaussFilter(self.deliver, self.time, self.entity, self.raw_features,
                                      self.stride, [self.scale], slope=self.slope, variance=self.variance) 
            self.data_dict = []
            self.all_features = self.filter.resultkeys()
            self.features = [feat for feat in self.all_features if not feat in [self.time, self.entity]]
        else:
            self.features = self.raw_features
            tmp = ([self.entity] if self.entity is not False else []) + self.features
            self.data_dict = self.df[tmp].to_dict(orient='records')
            self.all_features = [self.index] + tmp
            for i,d in enumerate(self.data_dict):
                d[self.index] = i
                key = (i, d[self.entity]) if self.entity is not False else (i,)
                self.lookupkey[key] = i

    def next(self):
        if self.data_dict is None:
            self.prepare_data()
        if self.time is not False and self.raw_data_index < len(self.raw_data_dict):
            while self.raw_data_index < len(self.raw_data_dict) and not self.data_index < len(self.data_dict):
                self.filter.handle(self.raw_data_dict[self.raw_data_index])
                self.raw_data_index += 1
        if not self.data_index < len(self.data_dict):
            return None
        else:
            dict_msg = self.data_dict[self.data_index]
            self.data_index += 1
            return dict_msg

    def all(self):
        if self.data_dict is None:
            self.prepare_data()
        if self.time is not False and self.raw_data_index < len(self.raw_data_dict):
            for vec in self.raw_data_dict[self.raw_data_index:]:
                self.filter.handle(vec)
            self.raw_data_index = len(self.raw_data_dict)
        self.data_index = len(self.data_dict)
        return self.data_dict

    def available(self):
        if self.data_dict is None:
            self.prepare_data()
        if self.time is not False and self.raw_data_index < len(self.raw_data_dict):
            while self.raw_data_index < len(self.raw_data_dict) and not self.data_index < len(self.data_dict):
                self.filter.handle(self.raw_data_dict[self.raw_data_index])
                self.raw_data_index += 1
        return self.data_index < len(self.data_dict)

    def len(self):
        return len(self.data_dict)

    def current_data(self):
        if self.horizon==None:
            if self.data_index == len(self.data_dict):
                return self.data_dict
            else:
                return self.data_dict[:self.data_index]
        else:
            return self.data_dict[max(0, self.data_index-self.horizon), self.data_index]

    def get_feature_names(self):
        return self.features

    def update(self, dictlst):
        keyattr = [self.index if self.time is False else self.time] + ([self.entity] if self.entity is not False else [])
        for ele in dictlst:
            key = tuple([ele[k] for k in keyattr])
            if key in self.lookupkey:
                self.data_dict[self.lookupkey[key]].update(ele)



## Repo functionality
#
# - set_orig_feature_names
# - add
# * update
# - cleanup
# * len
# * get_entities
# ? get
# ? get_keys
# * get_orig_feature_names
# ? get_values
# ? get_time_now
# ? get_time_window_seconds
    
#def main():
#    pass
#
#if __name__ == "__main__":
#    main()
