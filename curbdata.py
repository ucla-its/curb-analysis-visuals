import numpy as np
import pandas as pd
import datetime as dt

import sys
import operator
import copy

import fastparquet
import snappy
from utilities import color_blind_10
import seaborn as sns
import ipywidgets as widgets

class CurbData:
    """Stores curb utilization data and associated functions"""
    def __init__(self, timestamped_df, format_dict):
        self.format = format_dict
        self.df_all = timestamped_df
        self.df_subset = self.df_filtered = self.__time_filter__()
        self.subset_type = 'All Data'
        self.subset_duration = max(self.df_subset.index) - min(self.df_subset.index)
        
    def violator_timestamp_table(self):
        #TODO fixed bug, but still a cleanup candidate
        '''Generates time-indexed df counting number and type of violators present.'''
        if self.df_subset.shape[0] == 0:
            print('Unable to calculate violators,',
                    '\ntime selection does not include enforcement interval ')
            return
        lz_spaces = self.df_subset[~self.df_subset.loc[
            :, self.format['location column']].isin(self.format['unrestricted spaces'])]
        outputdf = pd.DataFrame()
    
        def counts_to_df(counts_series, viol_label):
            new_column = viol_label
            try:
                df = pd.DataFrame(counts_series[:, viol_label])
                df = df.rename({self.format['violator column']: viol_label}, axis=1)
            except:
                print('index failed', 
                      len(mask), 
                      len(counts_series.loc[:, viol_label]))
                print('lbl:{}, value:{}'.format(viol_label, viol_value))
            return df

        df_grouped = lz_spaces.groupby(level=0)
        viol_counts = df_grouped[self.format['violator column']].value_counts()
        viol_labels = lz_spaces[self.format['violator column']].unique()
        for label in viol_labels:
            countdf = counts_to_df(counts_series=viol_counts, viol_label=label)
            if type(countdf) == pd.core.frame.DataFrame:
                outputdf = outputdf.join(countdf, how='outer')

        violdf = outputdf[outputdf.columns[outputdf.columns ==
            self.format['violator column']]].fillna(value=0)
        blockdf = outputdf[outputdf.columns[outputdf.columns.str.contains(
            'Blocking')]].fillna(value=0)
        outputdf['Total Violators'] = violdf.sum(axis=1)
        outputdf['Total Blockers'] = blockdf.sum(axis=1)

        outputdf['Blockers Less Violators'] = outputdf[
                'Total Blockers'] - outputdf['Total Violators']
        outputdf['Any Violator'] = outputdf['Total Violators'].apply(
                lambda x: 1 if x > 0 else 0)
        outputdf['Any Blocking'] = outputdf['Total Blockers'].apply(
                lambda x: 1 if x > 0 else 0)

        return outputdf

    def blockers_exceed_viol(self):
        '''Returns a value (in minutes) of the amount of time that the number of blockers
        exceeds the number of violators. A relatively high number may suggest the zone is
        too small'''
        v_table = self.violator_timestamp_table()
        if type(v_table) == type(None):
            return
        exceed_min = v_table[v_table['Blockers Less Violators'] > 0].shape[0] / 60
        total_min = self.subset_duration.seconds / 60
        return {'blockers_exceed':exceed_min, 'total_minutes':total_min}

    def __time_filter__(self):
        '''filters df to enforcement interval provided in format (as 24-hr time hh:mm)'''
        
        self.df_all.sort_index(level='Timestamp', inplace=True)
        enf_start = self.format['enf start/end'][0].split(':')
        after_start = self.df_all.index.time > dt.time(int(enf_start[0]), int(enf_start[1]))
        df_after_st = self.df_all.loc[after_start]
        enf_end = self.format['enf start/end'][1].split(':')
        before_end = df_after_st.index.time < dt.time(int(enf_end[0]), int(enf_end[1]))
        df_in_interval = df_after_st.loc[before_end]
        
        return df_in_interval
    
    def blk_viol_times(self, condition=None):
        '''returns dict of total times any blocker or violator present in interval'''
        df = self.violator_timestamp_table()
        if condition == 'Blocking':
            df = df[df['Any Blocking'] == 1]
        elif condition == 'Violator':
            df = df[df['Any Violator'] == 1]
            
        block_sec = df.sum()['Any Blocking']
        block_min = int(block_sec / 60)
        viol_sec = df.sum()['Any Violator']
        viol_min = int(viol_sec / 60)
        return {'block_sec': block_sec, 'block_min': block_min, 
                'viol_sec': viol_sec, 'viol_min': viol_min}

    def conditional(self, condition):
        '''prints and returns observed conditional of either 'Blocking' or 'Violator'''
        times = self.blk_viol_times(condition=condition)
        if condition == 'Blocking':
            #further abstract text?
            print(("Out of the {} minutes the bike lane was blocked in the study period,"
                   " at least one violator was parked in the loading zone for"
                   " {} minutes ({} % of the time!)".format(
                       times['block_min'], times['viol_min'], 
                       int((times['viol_min']/times['block_min'])*100 ))))     
        elif condition == 'Violator':
            print(("Out of the {} minutes that at least one violator was" 
                    " parked in the loading zone,"
                   " the bike lane was blocked for {} minutes ({} % of the time!)".format(
                       times['viol_min'], times['block_min'], 
                       int((times['block_min']/times['viol_min'])*100 ))))
        return times
    
    def subset(self, weekday=None, timestamps=None, verbose=False):
        '''returns new CurbData object containing specified subset. Can specify as timestamps
        = ('yyyy-mm-dd hh:mm:ss', 'yyyy-mm-dd hh:mm:ss'), or weekday = any day of week, 
        Weekdays, Weekends, All Data.'''
    # NOTE currently subset of subset can't expand. Could change by reloading df_filtered 
        if verbose:
            print('Updating selection...', end='')
        def from_weekday(weekday):
            '''returns df_subset of all matching weekday
            (or, weekdays and weekends)
            also, appends each single day df to self.oneday_dfs
            '''
            if weekday not in ['Weekdays', 'Weekends']:
                df_subset = self.df_subset[
                        self.df_subset.index.strftime("%A") == weekday]
            elif weekday == 'Weekdays':
                df_subset = self.df_subset[self.df_subset.index.strftime(
                    "%w").astype(int).isin(list(np.arange(1, 6)))]
            elif weekday == 'Weekends':
                df_subset = self.df_subset[~self.df_subset.index.strftime(
                    "%w").astype(int).isin(list(np.arange(1, 6)))]
            return df_subset
        
        def from_timestamps(timestamps):
            '''simply returns df_subset between two timestamps
            opt, if needed split single days if we accept
            input > 1day...
            '''
            df_subset = self.df_subset[self.df_subset.index > timestamps[0]]
            df_subset = df_subset[df_subset.index < timestamps[1]]
            return df_subset
        
        if weekday:
            if weekday not in ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday', 'Saturday', 'Weekdays', 'Weekends']:
                raise Exception('Must be a day of week, "Weekdays", or "Weekends"')
                return
            else:
                df_subset = from_weekday(weekday)
                subset_type = weekday
        elif timestamps:
            if timestamps[0] == timestamps[1]:
                raise Exception('Start and end times must be different')
            if len(timestamps) != 2:
                raise Exception('Timestamps must be a list/tuple of 2 datetime objects')
                return
            else:
                df_subset = from_timestamps(timestamps)  
                subset_type = timestamps
        else:
            raise Exception('Must select a valid subset interval')
            return

        subset = copy.copy(self)
        subset.df_subset = df_subset
        subset.subset_type = subset_type
        subset.subset_duration = max(df_subset.index) - min(df_subset.index)
        if verbose:
            print('Done!')
        return subset

    def aggregate_activities(self, percent=False, lz_only=True):
        '''return counts and total time for each type of activity in subset interval
        also percent blocking, percent viol presence 
        '''
        #assert by == None or 'day' or 'hour'
        ##may implement if useful
        
        if lz_only:
            lz_df = self.df_subset[~self.df_subset.loc[
            :, self.format['location column']].isin(self.format['unrestricted spaces'])]
        first = lz_df.drop_duplicates(keep='first')
        last = lz_df.drop_duplicates(
            keep='last').reset_index().rename(columns={'Timestamp':'last_in_interval'})
        last = last[['Activity Id', 'last_in_interval']]
        joined = first.join(last.set_index('Activity Id'), on='Activity Id')
        joined['duration_in_interval'] = joined['last_in_interval'] - joined.index
        grouped = joined.groupby(self.format['violator column'])
        agged = grouped.agg({'duration_in_interval':'sum',
                    'Bikeway Users Displaced': 'sum'})
        counts = grouped.count()['Activity Id']
        agged = agged.join(counts).rename(columns={'Activity Id':'Activity Count',
                                                   'duration_in_interval':'Total Time',
                                                  })
        agged.index.rename('Violator Classification', inplace=True)
        cols = ['Total Time', 'Activity Count', 'Bikeway Users Displaced']
        agged = agged[cols]
        if percent:
            agged[cols] = agged[cols] = agged[cols] / agged[cols].sum()
            agged.rename(columns={'Total Time':'Percent Time',
                                'Activity Count':'Percent Activities',
                                'Bikeway Users Displaced':'Percent Bikeway Users Displaced'},
                                inplace=True)
        return agged

    def download_selection(self, download_all=False):
        """Generates downloadable CSV from input df
        Total copy/paste job from 
        https://stackoverflow.com/questions/31893930/download-csv-from-an-ipython-notebook
        Javascript allows for client-side CSV generation, avoids creating server-side CSV
        for each request
        Tends to generate two duplicate downloads on Firefox, but one on Safari. Have yet 
        to test with Chrome. Likely ipython/Jupyter/browser quirk. 
        """
        if download_all:
            download_df = self.df_all
        else:
            download_df = self.df_subset
        from IPython.display import Javascript
        js_download = """
        var csv = '%s';
    
        var filename = 'CurbDataExport.csv';
        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        if (navigator.msSaveBlob) { // IE 10+
            navigator.msSaveBlob(blob, filename);
        } else {
            var link = document.createElement("a");
            if (link.download !== undefined) { // feature detection
                // Browsers that support HTML5 download attribute
                var url = URL.createObjectURL(blob);
                link.setAttribute("href", url);
                link.setAttribute("download", filename);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
        """ % (download_df.drop_duplicates()
                .to_csv(index=False).replace('\n','\\n').replace("'","\'"))
    #     time.sleep(.5)
        return Javascript(js_download)


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

class CurbDataPlotter(CurbData):
    def __init__(self, timestamped_df, format_dict):
        super().__init__(timestamped_df, format_dict)
        #color-blind safe 10-color categorical
        self.palette = self.__make_palette__()

    #DONE more color-blind safe colors to break out TNC/CNS blocking?
    #Also further abstraction concurrent with timestamper improvements.
    def __make_palette__(self):
        viol_types = np.sort(self.df_filtered[self.format['violator column']].unique())
        #assumes violator types from current timestamper

        color_list = [color_blind_10['gray4'], color_blind_10['gray3'],
                     color_blind_10['gray2'], color_blind_10['sky'],
                     color_blind_10['blue'], color_blind_10['navy'],
                     color_blind_10['brick'], color_blind_10['orange'],
                     color_blind_10['peach'], color_blind_10['gray1']]
        #complex tuple/list comp since Seaborn wants rgb values 0-1...
        palette = dict(zip(viol_types, sns.color_palette(
            list([tuple(value/255 for value in color) for color in color_list]))))
        return palette

    def time_occ_plot(self, save=False):
        """Plots space occupancy and bike lane blocking over time in the selected interval. 
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.cbook as cbook
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        #return int for size based on plot duration
        def best_size(duration_min):
            if duration_min < 2:
                return 7
            elif duration_min < 5:
                return 6
            elif duration_min < 15:
                return 5
            elif duration_min < 60:
                return 3
            else:
                return 2
        #filter to only key spaces to avoid cluttered plot
        key_space_df = self.df_subset.loc[self.df_subset[
            self.format['location column']].isin(self.format['key spaces'])] 
        #order locations
        key_space_df = key_space_df.sort_values(by=[self.format['location column']],)
        key_space_df['Time'] = key_space_df.index
        start = self.df_subset.index.min().to_pydatetime()
        end = self.df_subset.index.max().to_pydatetime()
        duration_min = ((end - start).seconds) / 60
        if duration_min > 60*24:
            raise Exception('Time/Occcupancy Plot only for selections under 24 hours.')
            return
        
        fig, ax = plt.subplots()
        #plot using Seaborn strip plot, set x-axis range
        #hue based on 'Violator' values, which include TNC/CNS status 
        ax = sns.stripplot(x="Time", y=self.format['location column'],
            hue=self.format['violator column'], palette = self.palette,
            data=key_space_df, size = best_size(duration_min), jitter=False)
        ax.set_xlim([start, end])
        #move legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #title plot with date
        ax.set_title(key_space_df['Begin Date'][0])
        #format times on x-axis to readable '9:30PM' format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%-I:%M %p'))
        fig.autofmt_xdate()
        #set attribute to enable downloading
        self.plot = fig
        plt.show()
        return ax
    
    def time_activity_plot(self, percent=True):
        '''Generate plot aggregating the total number of activities and amount of time
        for each user/violator type in subset interval. Can specify percent=false for
        absolute counts instead of stacked percentage.'''
        import matplotlib.pyplot as plt
        agged = self.aggregate_activities(percent)
        subset_type = self.subset_type
        st_dt = min(self.df_subset.index) 
        end_dt = max(self.df_subset.index)
        span = ''
        if type(self.subset_type) == str:
            st_month_day = st_dt.strftime('%m/%d')
            end_month_day = end_dt.strftime('%m/%d/%Y')
            plural = ''
            if self.subset_type[-1] == 'y':
                plural = '(s)'
            span = f'{plural} {st_month_day} – {end_month_day}'
        else:
            subset_type = st_dt.strftime('%m/%d/%Y %I:%M%p–') + end_dt.strftime('%I:%M%p')

        if not percent:
            with pd.option_context('mode.chained_assignment', None):
                df_to_plot = agged.loc[:, agged.columns != 'Bikeway Users Displaced']
                df_to_plot.loc[:,'Total Time'] = df_to_plot['Total Time'].apply(
                        lambda x: (x.seconds / 60) + (x.days * 24 * 60))
                df_to_plot = (df_to_plot
                        .rename(columns={'Total Time': 'Total Time (Minutes)'})
                        .transpose())
                fig, (ax1, ax2) = plt.subplots(1, 2)
                (df_to_plot.loc[['Total Time (Minutes)'],:]
                        .plot.bar(color=[self.palette.get(x) for x in df_to_plot.columns],
                    stacked=False, ax=ax1))
                (df_to_plot.loc[['Activity Count'],:]
                        .plot.bar(color=[self.palette.get(x) for x in df_to_plot.columns],
                    stacked=False, ax=ax2))

                fig.suptitle(f'Time and Activities by User/Violator Type, {subset_type}{span}')
                handles, labels = ax2.get_legend_handles_labels()
                ax1.get_legend().remove()
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.draw()
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
                #fig.legend(handles, labels, bbox_to_anchor=(1.05, 1))
                self.plot = fig
                return ax1, ax2
        if percent:
            df_to_plot = (agged
                    .apply(lambda x: x * 100)
                    .loc[:, agged.columns != 'Percent Bikeway Users Displaced']
                    .transpose())
            fig, ax = plt.subplots()
            df_to_plot.plot.bar(color=[self.palette.get(x) for x in df_to_plot.columns],
                stacked=True, ax=ax)

            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax.set_title(f'Time and Activities by User/Violator Type, {subset_type}{span}')
            plt.xticks(rotation=30)
            plt.show()
            self.plot = fig
            return ax

    def __make_widgets__(self):
        self.time_widget = widgets.SelectionRangeSlider(
                options=[0], continuous_update=False, layout={'width': '400px'})
        #TODO rewrite to use just index not Begin Date
        self.date_widget = widgets.Dropdown(options=self.df_subset['Begin Date'].unique())
        self.type_widget = widgets.ToggleButtons(
            options=['Detailled', 'Aggregate', 'Aggregate (percent)'],
            description='Plot Type',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['A detailled plot of space occupancy over time.\
                    Only available for single-day selections.', 
                      'A summary plot showing the total number of activities and total\
                              amount of time for each user type in the selection interval.',
                      'A summary plot showing the percentage of activities and percentage\
                              of time for each user type in the selection interval.'],
        )
        self.weekday_widget = widgets.Select(
            options=['Specific Date/Time', 'Monday', 'Tuesday', 'Wednesday', 
                    'Thursday', 'Friday', 'Saturday', 'Sunday', 'Weekdays', 'Weekends',
                    'All Data'],
            value='Specific Date/Time',
            style={'description_width': 'initial'},
            description='Aggregation Type:',
            disabled=False
        )
        self.download_button = widgets.Button(
                description='Download Data',
                button_style='',
                tooltip='Download Data',
                icon='download'
        )
        def update(*args):
            """allows date selection to define time selection range"""
            #TODO rewrite to use just index not Begin Date
            index = self.df_subset[
                    self.df_subset['Begin Date'] == self.date_widget.value].index
            minutes = index.strftime('%I:%M%p').unique()
            self.time_widget.options = minutes
            self.time_widget.value = (minutes[0], minutes[-1])
        self.date_widget.observe(update)

    def interactive_plot(self):
        '''Using ipython widgets, generates interactive plot enabling plot type and 
        subset specification.'''
        self.__make_widgets__()
        def __iplot__(self, plot_type, date, times, weekday):
            if plot_type == 'Detailled':
                self.weekday_widget.value = 'Specific Date/Time'
                self.weekday_widget.layout.display = 'none'
            else:
                self.weekday_widget.layout.display = 'initial'

            if weekday == 'Specific Date/Time':
                str_times = ['_'.join([date, time]) for time in times]
                timestamps = [dt.datetime.strptime(str_time, 
                    '%m/%d/%Y_%I:%M%p') for str_time in str_times]
                #try/except block only generates new subset when necessary (speedup)
                try:
                    if self.plot_selection.subset_type != timestamps:
                        self.plot_selection = self.subset(timestamps = timestamps)
                except AttributeError:
                        self.plot_selection = self.subset(timestamps = timestamps)
            elif weekday != 'All Data':
                try:
                    if self.plot_selection.subset_type != weekday:
                        self.plot_selection = self.subset(weekday = weekday, verbose = True)
                except AttributeError:
                        self.plot_selection = self.subset(weekday = weekday, verbose = True)
            elif weekday == 'All Data':
                self.plot_selection = self

            if plot_type == 'Detailled':
                self.plot_selection.time_occ_plot()
            elif plot_type == 'Aggregate':
                self.plot_selection.time_activity_plot(percent=False)
            elif plot_type == 'Aggregate (percent)':
                self.plot_selection.time_activity_plot(percent=True)

            from IPython.display import display, FileLink
            self.download_button.on_click(lambda x: 
                    display(self.plot_selection.download_selection()))
            display(self.download_button)
            self.plot_selection.plot.savefig('plot.png', bbox_inches = 'tight')
            display(FileLink('plot.png', result_html_prefix="Download plot: "))
        
        widgets.interact(lambda x, Date, Time, weekday: 
                __iplot__(self, x, Date, Time, weekday), 
                x = self.type_widget, Date = self.date_widget, Time = self.time_widget,
                weekday = self.weekday_widget)
