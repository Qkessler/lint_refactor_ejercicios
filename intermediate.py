from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import json
import io
import os
import re
from fastavro import writer, parse_schema
from azure.storage.blob import ContainerClient

from forma_LA.constants import (
    EVENTS_COLUMNS,
    GROUPING_VARS,
    GRID_WIDTH,
    INTERVAL_NEW_SESSION,
    OPENCONTENT_DOMAIN,
)
from forma_LA.container import Container
from forma_LA.schemes import intermediate_schemes
from forma_LA.aggregate import (
    AggregateRepository,
    AggregateAuthor,
    AggregateUnit,
    AggregateLTI,
)
from forma_LA.utilities import hold_same


class Intermediate(ABC):
    def __init__(self, container, path, events_df, backup_container):
        self.__intermediate = container.retrieve_blob(
            path, backup_container
        )

        self.events = events_df
        self.path = path


    @abstractmethod
    def prepare(self):
        self.__prepare_events_dates()

    def __prepare_events_dates(self):
        if hasattr(self, "first_event_date"):
            self.first_event_date = self.__intermediate["first_event_date"]
            self.last_event_date = self.__intermediate["last_event_date"]

    def __prepare_daily_visits(self):
        self.daily_visits = pd.DataFrame(
            self.__intermediate["daily_visits"],
            columns=["day", "num_visits", "num_visits_lti", "num_visits_open"],
            dtype=str,
        ).astype(
            {"num_visits": float, "num_visits_lti": float, "num_visits_open": float}
        )
        self.daily_visits["day"] = pd.to_datetime(
            self.daily_visits["day"], format="%Y-%m-%d"
        )

    def __prepare_top10_units(self):
        self.top10_units = pd.DataFrame(
            self.__intermediate["top10_units"],
            columns=[
                "unit",  # "url",# debe ser "unit"!, s
                "url",
                "current_title",
                "num_visits",
                "num_visits_lti",
                "num_visits_open",
            ],
        ).astype(
            {"num_visits": float, "num_visits_lti": float, "num_visits_open": float}
        )  # .rename(columns={"url": "unit"})

    def __prepare_num_visits(self):
        self.num_visits, self.num_visits_lti, self.num_visits_open = (
            (0, 0, 0)
            if not "num_visits" in self.__intermediate
            else (
                int(self.__intermediate["num_visits"]),
                int(self.__intermediate["num_visits_lti"]),
                int(self.__intermediate["num_visits_open"]),
            )
        )

    @abstractmethod
    def update(self):
        self.__update_events_dates()

    def __update_events_dates(self):
        date_time = datetime.fromtimestamp(self.events["timestamp"].iloc[0])
        self.first_event_date = date_time.strftime("%Y/%m/%d/%H/%M/%S")
        date_time = datetime.fromtimestamp(self.events["timestamp"].iloc[-1])
        self.last_event_date = date_time.strftime("%Y/%m/%d/%H/%M/%S")

    def __update_num_visits(self):
        logins = self.events.loc[self.events["type"] == "LoggedIn"]
        self.num_visits += logins.shape[0]
        self.num_visits_open += (logins["domain"] == OPENCONTENT_DOMAIN).sum()
        self.num_visits_lti += (logins["domain"] != OPENCONTENT_DOMAIN).sum()

    def __update_daily_visits(self):
        logins = self.events.loc[self.events["type"] == "LoggedIn"].copy()
        logins["num_visits_open"] = (logins["domain"] == OPENCONTENT_DOMAIN).astype(int)
        logins["num_visits_lti"] = (logins["domain"] != OPENCONTENT_DOMAIN).astype(int)

        new_daily_visits = logins.groupby("day")[
            ["num_visits_open", "num_visits_lti"]
        ].sum()
        new_daily_visits["num_visits"] = new_daily_visits.sum(axis=1)

        self.daily_visits = (
            pd.concat(
                [
                    self.daily_visits,
                    new_daily_visits.reset_index(),
                ]
            )
            .groupby("day")
            .sum()
            .reset_index()
        )

    def update_top10_units(self, intermediate_unit):
        self.top10_units.set_index(["unit"], inplace=True)

        unit = intermediate_unit.events["unit"].iloc[0]

        if unit in self.top10_units.index:
            url_top10 = self.top10_units.loc[unit, "url"]
            self.top10_units.drop(index=[unit], inplace=True)
        else:
            url_top10 = "" 

        url_list = [url_top10, *(intermediate_unit.events["url"].to_list())] 
        url_list.sort(reverse=True, key=len)
        url = url_list.pop(0)
        del url_list

        top10_units = pd.concat(
            [
                self.top10_units,
                pd.DataFrame(
                    {
                        "url": url,
                        "current_title": intermediate_unit.current_title,
                        "num_visits_lti": intermediate_unit.num_visits_lti,
                        "num_visits_open": intermediate_unit.num_visits_open,
                        "num_visits": intermediate_unit.num_visits,
                    },
                    index=[unit],
                ),
            ]
        )

        top10_units.index.name = "unit"

        top10_visits_index = top10_units.sort_values(
            "num_visits", ascending=False
        ).index[:10]

        top10_lti_index = top10_units.sort_values(
            "num_visits_lti", ascending=False
        ).index[:10]

        top10_open_index = top10_units.sort_values(
            "num_visits_open", ascending=False
        ).index[:10]

        top10_index = (
            top10_visits_index.append(top10_lti_index).append(top10_open_index)
        ).drop_duplicates()

        self.top10_units = (
            top10_units.loc[top10_index, :]
            .reset_index()
            .sort_values("num_visits", ascending=False)
        )

    @abstractmethod
    def as_dict(self):
        result = {
            "first_event_date": self.first_event_date,
            "last_event_date": self.last_event_date,
        }
        return result

    def __add_visits_dict(self):
        result = {
            "num_visits": str(self.num_visits),
            "num_visits_open": str(self.num_visits_open),
            "num_visits_lti": str(self.num_visits_lti),
            "daily_visits": self.daily_visits.astype(str).to_dict("records"),
        }
        return result

    @abstractmethod
    def upload(self):
        pass

    def __upload(self, container, path, schema):
        fo = io.BytesIO()
        writer(fo, parse_schema(schema), [self.as_dict()])
        ContainerClient.upload_blob(
            container.container, name=path, data=fo.getvalue(), overwrite=True
        )

    @abstractmethod
    def to_aggregate(self):
        pass

    def to_json(self, path):
        folder = re.sub("intermediate.avro", "", path)
        os.makedirs(folder, exist_ok=True)
        with open(path, "a") as jsonfile:
            json.dump(self.as_dict(), jsonfile)

    def get_dict_object(self):
        return self.__intermediate

    def remove_num_visits_col(self, objects):
        objects_dict = self.get_dict_object()
        for o in objects:
            for element in objects_dict[o]:
                del element["num_visits"]

    def change_url_to_unit(self, objects):
        objects_dict = self.get_dict_object()
        for o in objects:
            for element in objects_dict[o]:
                element["unit"] = element.pop("url")


    @abstractmethod
    def compare(self, intermediate):
        pass

    def __compare_num_visits(self, intermediate):
        dict_self = self.get_dict_object()
        dict_intermediate = intermediate.get_dict_object()

        comparison = dict()
        comparison["num_visits"] = (dict_self["num_visits"] == dict_intermediate["num_visits"])
        comparison["num_visits_open"] = (dict_self["num_visits_open"] == dict_intermediate["num_visits_open"])
        comparison["num_visits_lti"] = (dict_self["num_visits_lti"] == dict_intermediate["num_visits_lti"])
        comparison["daily_visits"] = (dict_self["daily_visits"] == dict_intermediate["daily_visits"])

        return comparison

class IntermediateRepository(Intermediate):
    def prepare(self):
        super().prepare()
        self._Intermediate__prepare_num_visits()
        self._Intermediate__prepare_daily_visits()
        self._Intermediate__prepare_top10_units()
        self.__prepare_top10_authors()

    def __prepare_top10_authors(self):
        self.top10_authors = pd.DataFrame(
            self._Intermediate__intermediate["top10_authors"],
            columns=["author", "num_visits", "num_visits_lti", "num_visits_open"],
        ).astype(
            {"num_visits": float, "num_visits_lti": float, "num_visits_open": float}
        )

    def update(self):
        super().update()
        self._Intermediate__update_num_visits()
        self._Intermediate__update_daily_visits()

    def update_top10_authors(self, intermediate_author):
        self.top10_authors.set_index("author", inplace=True)

        author = intermediate_author.author

        if author in self.top10_authors.index:
            self.top10_authors.drop(index=[author], inplace=True)

        top10_authors = pd.concat(
            [
                self.top10_authors,
                pd.DataFrame(
                    {
                        "num_visits_lti": intermediate_author.num_visits_lti,
                        "num_visits_open": intermediate_author.num_visits_open,
                        "num_visits": intermediate_author.num_visits,
                    },
                    index=[author],
                ),
            ]
        )

        top10_authors.index.name = "author"

        top10_visits_index = top10_authors.sort_values(
            "num_visits", ascending=False
        ).index[:10]

        top10_lti_index = top10_authors.sort_values(
            "num_visits_lti", ascending=False
        ).index[:10]

        top10_open_index = top10_authors.sort_values(
            "num_visits_open", ascending=False
        ).index[:10]

        top10_index = (
            top10_visits_index.append(top10_lti_index).append(top10_open_index)
        ).drop_duplicates()

        self.top10_authors = (
            top10_authors.loc[top10_index, :]
            .reset_index()
            .sort_values("num_visits", ascending=False)
        )

    def as_dict(self):
        result = {
            **super().as_dict(),
            **self._Intermediate__add_visits_dict(),
            "top10_units": self.top10_units.astype(str).to_dict("records"),
            "top10_authors": self.top10_authors.astype(str).to_dict("records"),
        }
        return result

    def upload(self, container):
        super().upload()
        schema = intermediate_schemes["repository"]
        self._Intermediate__upload(container, self.path, schema)

    def to_aggregate(self):
        super().to_aggregate()
        aggregate = AggregateRepository()
        aggregate.add_path(self)
        aggregate.add_visits(self)
        aggregate.extract_top10_units(self)
        aggregate.extract_top10_authors(self)
        aggregate.compute_open_indicators()
        aggregate.aggregate_dict = aggregate.as_dict()
        return aggregate

    def compare(self, intermediate):
        dict_self = self.get_dict_object()
        dict_intermediate = intermediate.get_dict_object()

        super().compare(intermediate)
        comparison = self._Intermediate__compare_num_visits(intermediate)
        comparison = {**comparison,
                      "top10_units": dict_self["top10_units"] == dict_intermediate["top10_units"],
                      "top10_authors": dict_self["top10_authors"] == dict_intermediate["top10_authors"],
                      }
        return comparison



class IntermediateAuthor(Intermediate):
    def prepare(self):
        super().prepare()
        self._Intermediate__prepare_num_visits()
        self._Intermediate__prepare_daily_visits()
        self._Intermediate__prepare_top10_units()
        self.author = self.events["author"].iloc[0]

    def update(self):
        super().update()
        self._Intermediate__update_num_visits()
        self._Intermediate__update_daily_visits()

    def as_dict(self):
        result = {
            **super().as_dict(),
            **self._Intermediate__add_visits_dict(),
            "top10_units": self.top10_units.astype(str).to_dict("records"),
        }
        return result

    def upload(self, container):
        super().upload()
        schema = intermediate_schemes["author"]
        self._Intermediate__upload(container, self.path, schema)

    def to_aggregate(self):
        super().to_aggregate()
        aggregate = AggregateAuthor()
        aggregate.add_path(self)
        aggregate.add_visits(self)
        aggregate.extract_top10_units(self)
        return aggregate

    def compare(self, intermediate):
        dict_self = self.get_dict_object()
        dict_intermediate = intermediate.get_dict_object()

        super().compare(intermediate)
        comparison = self._Intermediate__compare_num_visits(intermediate)
        comparison = {**comparison,
                      "top10_units": dict_self["top10_units"] == dict_intermediate["top10_units"],
                      }
        return comparison

class IntermediateUnit(Intermediate):
    def prepare(self):
        super().prepare()
        self._Intermediate__prepare_num_visits()
        self._Intermediate__prepare_daily_visits()
        self.author = self.events["author"].iloc[0]

    def update(self):
        super().update()
        self._Intermediate__update_num_visits()
        self._Intermediate__update_daily_visits()
        self.__update_current_title()

    def __update_current_title(self):
        self.current_title = self.events["title"].iloc[-1]

    def as_dict(self):
        result = {
            **super().as_dict(),
            **self._Intermediate__add_visits_dict(),
            "current_title": self.current_title,
        }
        return result

    def upload(self, container):
        super().upload()
        schema = intermediate_schemes["unit"]
        self._Intermediate__upload(container, self.path, schema)

    def to_aggregate(self):
        super().to_aggregate()
        aggregate = AggregateUnit()
        aggregate.add_path(self)
        aggregate.add_visits(self)
        aggregate.current_title = self.current_title
        return aggregate

    def compare(self, intermediate):
        dict_self = self.get_dict_object()
        dict_intermediate = intermediate.get_dict_object()

        super().compare(intermediate)
        comparison = self._Intermediate__compare_num_visits(intermediate)
        comparison = {**comparison,
                      "current_title": dict_self["current_title"] == dict_intermediate["current_title"],
                      }
        return comparison

class IntermediateLTI(Intermediate):
    def prepare(self):
        super().prepare()
        self.__prepare_last_event()
        self.__prepare_visited_units()
        self.__prepare_users()
        self.__prepare_user_progress()
        self.__prepare_daily_effort()
        self.__prepare_video_last_event()
        self.__prepare_video_borders()
        self.__prepare_video_current_title()
        self.__prepare_video_current_duration()
        self.__prepare_events()
        self.__prepare_video_events()


    def compare(self, intermediate):
        dict_self = self.get_dict_object()
        dict_intermediate = intermediate.get_dict_object()

        comparison = dict()
        comparison["last_event"] = (dict_self["last_event"] == dict_intermediate["last_event"])
        comparison["visited_units"] = (dict_self["visited_units"] == dict_intermediate["visited_units"])
        comparison["users"] = (dict_self["users"] == dict_intermediate["users"])
        comparison["user_progress"] = (dict_self["user_progress"] == dict_intermediate["user_progress"])
        comparison["daily_effort"] = (dict_self["daily_effort"] == dict_intermediate["daily_effort"])
        # comparison["video_last_event"] = (dict_self["video_last_event"] == dict_intermediate["video_last_event"])
        comparison["video_borders"] = (dict_self["video_borders"] == dict_intermediate["video_borders"])
        comparison["video_current_title"] = (dict_self["video_current_title"] == dict_intermediate["video_current_title"])
        comparison["video_current_duration"] = (dict_self["video_current_duration"] == dict_intermediate["video_current_duration"])

        return comparison

    def __prepare_last_event(self):
        self.last_event = pd.DataFrame(
            self._Intermediate__intermediate["last_event"],
            columns=[*EVENTS_COLUMNS, "day", "author", "unit", "time_spent"],
            dtype=str,
        ).astype({"percentage": float, "timestamp": float, "time_spent": float})
        self.last_event["day"] = pd.to_datetime(
            self.last_event["day"], format="%Y-%m-%d"
        )

    def __prepare_visited_units(self):
        self.visited_units = pd.DataFrame(
            self._Intermediate__intermediate["visited_units"],
            columns=[
                "domain",
                "course",
                "activity",
                "unit",
                "unit_type",
                "current_title",
            ],
            dtype=str,
        )

    def __prepare_users(self):
        self.users = pd.DataFrame(
            self._Intermediate__intermediate["users"],
            columns=[
                "domain",
                "course",
                "user",
                "name",
                "email",
                "profile",
            ],
            dtype=str,
        )

    def __prepare_user_progress(self):
        self.user_progress = pd.DataFrame(
            self._Intermediate__intermediate["user_progress"],
            columns=[
                "domain",
                "course",
                "activity",
                "unit",
                "unit_type",
                "user",
                "percentage",
                "time_spent",
            ],
            dtype=str,
        ).astype({"percentage": float, "time_spent": float})

    def __prepare_daily_effort(self):
        self.daily_effort = pd.DataFrame(
            self._Intermediate__intermediate["daily_effort"],
            columns=["domain", "course", "user", "day", "time_spent"],
            dtype=str,
        ).astype({"time_spent": float})
        self.daily_effort["day"] = pd.to_datetime(
            self.daily_effort["day"], format="%Y-%m-%d"
        )

    def __prepare_video_last_event(self):
        self.video_last_event = pd.DataFrame(
            self._Intermediate__intermediate["video_last_event"],
            columns=[
                *EVENTS_COLUMNS,
                "author",
                "unit",
                "day",
                "time_spent",
                "position",
                "action_type",
                "duration",
            ],
            dtype=str,
        ).astype(
            {
                "percentage": float,
                "timestamp": float,
                "time_spent": float,
                "position": float,
                "duration": float,
            }
        )

    def __prepare_video_borders(self):
        self.video_borders = pd.DataFrame(
            self._Intermediate__intermediate["video_borders"],
            columns=[*GROUPING_VARS, "element", "start", "end"],
            dtype=str,
        ).astype({"start": float, "end": float})

    def __prepare_video_current_title(self):
        self.video_current_title = pd.DataFrame(
            self._Intermediate__intermediate["video_current_title"],
            columns=[
                "domain",
                "course",
                "activity",
                "unit",
                "unit_type",
                "element",
                "description",
            ],
            dtype=str,
        )

    def __prepare_video_current_duration(self):
        video_current_duration = pd.DataFrame(
            self._Intermediate__intermediate["video_current_duration"],
            columns=[
                "domain",
                "course",
                "activity",
                "unit",
                "unit_type",
                "element",
                "duration_freq",
                "duration",
            ],
            dtype=str,
        )
        if video_current_duration.shape[0] > 0:
            video_current_duration["duration_freq"] = video_current_duration[
                "duration_freq"
            ].apply(lambda x: Counter(json.loads(x)))
        self.video_current_duration = video_current_duration

    def __prepare_events(self):
        events_columns = self.events.columns.to_list()
        
        self.__add_timestamp_last_event()
        
        self.__add_session()
        
        self.__add_time_spent()
        
        self.events = self.events.loc[:, [*events_columns, "time_spent"]]

    def __add_timestamp_last_event(self):

        last_event = self.last_event.copy()
        last_event["is_last_event"] = True

        self.events["is_last_event"] = False
        self.events["time_spent"] = np.nan
        self.events = pd.concat(
            [last_event, self.events],
            ignore_index=True
        )
        

    def __add_session(self):
        self.events["timestamp_previous_event"] = self.events.groupby(GROUPING_VARS)[
            "timestamp"
        ].shift()
        self.events["time_spent_previous_event"] = self.events.groupby(GROUPING_VARS)[
            "time_spent"
        ].shift()

        self.events["begins_session"] = ~(self.events["type"] == "LoggedOut") & (
            (self.events["type"] == "LoggedIn")
            | self.events["timestamp"].isna()
            | (
                (self.events["timestamp"] - self.events["timestamp_previous_event"])
                >= INTERVAL_NEW_SESSION
            )
        )

        self.events = self.events[~self.events["is_last_event"]]

        self.events["session"] = self.events.groupby(GROUPING_VARS)[
            "begins_session"
        ].cumsum()

        ## We may find that session = 0, which may correspond to a split
        ## session: when the batch of events was constructed, it did split a
        ## session. We then do as if in the last event of the first part
        ## of the split session (available through last_event), there is a
        ## logged out, logged in simultaneously. We therefore add, for
        ## those session = 0 a "fake' additional event which is a kind of
        ## duplicate of the event that was the last one (in last_event).
        # -----------------------------------------------------

        if hasattr(self, "last_event") and ((self.events["session"] == 0).sum() > 0):
            last_event = self.last_event
            additional_vars = ["email", "name", "profile", "title", "activity_title"]
            select_cols = [*GROUPING_VARS,  *additional_vars]
            new_events_session_0 = self.events.loc[
                self.events["session"] == 0, select_cols
            ].copy()
            new_events_session_0.drop_duplicates(inplace=True)

            # We only keep the first row of those events, in the unlikely case that
            # email has changed or name or profile

            new_events_session_0 = (
                new_events_session_0.groupby(GROUPING_VARS).last().reset_index()
            )
            new_events_session_0 = pd.merge(
                last_event.drop(columns=additional_vars),
                new_events_session_0,
                on=GROUPING_VARS,
            )

            # Notice that events in session 0 always have an associated row in
            # last_event. Otherwise, they would have begin_session = True

            new_events_session_0["begins_session"] = True
            new_events_session_0["is_last_event"] = True
            new_events_session_0["time_spent_previous_event"] = new_events_session_0[
                "time_spent"
            ]

            self.events = pd.concat(
                [new_events_session_0, self.events], ignore_index=True
            )

            self.events["session"] = self.events.groupby(GROUPING_VARS)[
                "begins_session"
            ].cumsum()

    def __add_time_spent(self):
        self.events["time_spent_increment"] = self.events.groupby(GROUPING_VARS)[
            "timestamp"
        ].diff().fillna(0) * (~self.events["begins_session"]) + self.events[
            "time_spent_previous_event"
        ].fillna(
            0
        ) * (
            self.events["begins_session"] & (self.events["session"] == 1)
        )

        self.events["time_spent"] = self.events.groupby(GROUPING_VARS)[
            "time_spent_increment"
        ].cumsum()

    def __prepare_video_events(self):
        video_events = self.events.loc[
            self.events["notes"].str.match("{ 'video' :"), :
        ].copy()
        if video_events.shape[0] == 0:
            self.video_events = None
        else:
            self.video_events = video_events
            self.__expand_video_notes()
            self.__round_time(self.video_events, ["duration"])

    def __expand_video_notes(self):
        notes = self.video_events["notes"].str.replace("'", '"')
        notes_list = [json.loads(n)["video"][0] for n in notes]

        self.video_events.loc[:, ["action_type", "Ranges", "duration"]] = (
            pd.DataFrame(notes_list, index=self.video_events.index)
            .rename(columns={"Type": "action_type", "Duration": "duration"})
            .astype({"duration": float})
        )

    def __round_time(self, dataframe, variables):
        for column in variables:
            dataframe[column] = np.round(dataframe[column] / GRID_WIDTH) * GRID_WIDTH

    # ------------------------------------------------------------------------------------------#

    def update(self):
        super().update()
        self.__update_last_event()
        self.__update_visited_units()
        self.__update_users()
        self.__update_user_progress()
        self.__update_daily_effort()
        self.__update_video_last_event()
        self.__update_video_borders()
        self.__update_video_current_title()
        self.__update_video_current_duration()

    def __update_last_event(self):
        new_last_events = (
            pd.concat([self.last_event, self.events])
            .groupby(GROUPING_VARS)
            .last()
            .reset_index()
        )

        self.last_event = new_last_events

    def __update_visited_units(self):
        grouping_units = GROUPING_VARS.copy()
        grouping_units.remove("user")

        new_visited_units = (
            pd.concat(
                [
                    self.visited_units,
                    (
                        self.events.loc[:, [*grouping_units, "activity_title"]].rename(
                            columns={"activity_title": "current_title"}
                        )
                    ),
                ]
            )
            .groupby(grouping_units)
            .last()
            .reset_index()
        )

        self.__spot_duplicated_titles(new_visited_units)

        self.visited_units = new_visited_units

    def __spot_duplicated_titles(self, dataframe):
        duplicated_titles = True
        while duplicated_titles:
            dataframe["title_occurrence"] = (
                dataframe["current_title"]
                .duplicated()
                .groupby(dataframe["current_title"])
                .cumsum()
            )
            duplicated_titles = (dataframe["title_occurrence"] > 0).any()
            if duplicated_titles:
                dataframe["current_title"] = dataframe["current_title"].where(
                    dataframe["title_occurrence"] == 0,
                    dataframe["current_title"]
                    + " "
                    + dataframe["title_occurrence"].astype(str)
                    + "/duplicated title",
                )

        dataframe.drop(columns=["title_occurrence"], inplace=True)

    def __update_users(self):
        grouping_users = ["domain", "course", "user"]
        new_users = (
            pd.concat(
                [
                    self.users,
                    self.events.loc[:, grouping_users + ["name", "email", "profile"]],
                ]
            )
            .groupby(grouping_users)
            .last()
            .reset_index()
        )
        self.users = new_users

    def __update_user_progress(self):
        new_user_progress = (
            self.events.loc[:, GROUPING_VARS + ["percentage", "time_spent"]]
            .groupby(GROUPING_VARS)
            .last()
            .reset_index()
        )
        new_user_progress = (
            pd.concat([self.user_progress, new_user_progress])
            .groupby(GROUPING_VARS)
            .last()
            .reset_index()
        )
        self.user_progress = new_user_progress

    def __update_daily_effort(self):
        grouping_vars = [*GROUPING_VARS, "day"]
        grouped_new_events = self.events.groupby(grouping_vars)
        daily_effort_activity = (
            grouped_new_events["time_spent"].max()
            - grouped_new_events["time_spent"].min()
        ).reset_index()
        grouping_vars.remove("unit")
        grouping_vars.remove("activity")
        grouping_vars.remove("unit_type")
        new_daily_effort = (
            daily_effort_activity.groupby(grouping_vars)["time_spent"]
            .sum()
            .reset_index()
        )
        new_daily_effort = (
            pd.concat([self.daily_effort, new_daily_effort])
            .groupby(grouping_vars)
            .sum()
            .reset_index()
        )
        self.daily_effort = new_daily_effort

    def __update_video_last_event(self):
        if self.video_events is not None:
            new_video_last_event = (
                pd.concat([self.video_last_event, self.video_events])
                .groupby([*GROUPING_VARS, "element"])
                .last()
                .reset_index()
            )
            self.video_last_event = new_video_last_event

    def __update_video_borders(self):
        if self.video_events is not None:
            new_borders = self.__get_borders()
            self.video_borders = pd.concat([self.video_borders, new_borders])

    def __get_borders(self):
        ranges_df = self.video_events["Ranges"].apply(lambda x: pd.DataFrame(x))

        ranges = pd.concat([df for df in ranges_df])
        ranges.index = ranges_df.index[(ranges.index == 0).cumsum() - 1]
        borders = pd.concat(
            [self.video_events[[*GROUPING_VARS, "element"]], ranges], axis=1
        )
        self.__round_time(borders, ["start", "end"])
        return borders

    def __update_video_current_title(self):
        if self.video_events is not None:
            grouping_vars = [*GROUPING_VARS, "element"]
            grouping_vars.remove("user")
            updated_title = (
                pd.concat(
                    [
                        self.video_current_title[[*grouping_vars, "description"]],
                        self.video_events[[*grouping_vars, "description"]],
                    ]
                )
                .groupby(grouping_vars)["description"]
                .last()
            )
            self.video_current_title = updated_title.reset_index()

    def __update_video_current_duration(self):
        if self.video_events is None:
            self.video_current_duration["duration_freq"] = self.video_current_duration[
                "duration_freq"
            ].apply(lambda x: json.dumps(x))
        else:
            grouping_vars = [*GROUPING_VARS, "element"]
            grouping_vars.remove("user")

            new_duration = self.video_events.groupby(grouping_vars).apply(
                lambda x: Counter(x["duration"].astype(str))
                )
            new_duration.name = "new_duration_freq"

            updated_duration = pd.concat(
                [
                    self.video_current_duration.set_index(grouping_vars),
                    new_duration,
                ],
                axis=1,
            )
            updated_duration["new_duration_freq"] = updated_duration[
                "new_duration_freq"
            ].where(~updated_duration["new_duration_freq"].isna(), Counter(dict()))
            updated_duration["duration_freq"] = updated_duration[
                "new_duration_freq"
            ].where(
                updated_duration["duration_freq"].isna(),
                updated_duration["duration_freq"]
                + updated_duration["new_duration_freq"],
            )
            updated_duration["duration"] = updated_duration["duration_freq"].apply(
                lambda x: x.most_common(1)[0][0]
            )
            updated_duration["duration_freq"] = updated_duration["duration_freq"].apply(
                lambda x: json.dumps(x)
            )
            updated_duration.reset_index(inplace=True)
            self.video_current_duration = updated_duration.drop(
                columns="new_duration_freq"
            )

    def as_dict(self):
        result = {
            **super().as_dict(),
            "last_event": self.last_event.astype(str).to_dict("records"),
            "visited_units": self.visited_units.astype(str).to_dict("records"),
            "users": self.users.astype(str).to_dict("records"),
            "user_progress": self.user_progress.astype(str).to_dict("records"),
            "daily_effort": self.daily_effort.astype(str).to_dict("records"),
            "video_last_event": self.video_last_event.astype(str).to_dict("records"),
            "video_borders": self.video_borders.astype(str).to_dict("records"),
            "video_current_title": self.video_current_title.astype(str).to_dict(
                "records"
            ),
            "video_current_duration": self.video_current_duration.astype(str).to_dict(
                "records"
            ),
        }
        return result

    def upload(self, container):
        super().upload()
        schema = intermediate_schemes["lti"]
        self._Intermediate__upload(container, self.path, schema)

    def to_aggregate(self):
        super().to_aggregate()
        aggregate = AggregateLTI()

        aggregate.add_path(self)
        aggregate.number_visited_units = self.visited_units.shape[0]
        aggregate.number_users = self.users.shape[0]
        aggregate.users = self.users
        aggregate.user_progress = self.__build_user_progress()
        aggregate.visited_completed_units = self.__build_visited_completed_units()
        aggregate.build_time_percentage_user_wide()
        aggregate.daily_effort = self.__build_daily_effort()
        aggregate.video_viz_cohort = self.__build_video_viz()

        return aggregate

    def __build_user_progress(self):
        user_progress = pd.merge(self.user_progress, self.visited_units)
        user_progress = pd.merge(user_progress, self.users).drop(columns="profile")
        user_progress["time_spent"] = pd.to_datetime(
            user_progress["time_spent"].astype(float), unit="s"
        ).dt.strftime("%H:%M:%S")
        return user_progress

    def __build_visited_completed_units(self):
        user_progress = self.user_progress

        grouping_vars = GROUPING_VARS.copy()
        grouping_vars.remove("user")

        cross_users_units = pd.merge(
            user_progress.loc[:, grouping_vars].drop_duplicates(),
            pd.DataFrame(user_progress["user"].unique(), columns=["user"]),
            how="cross",
        )

        completed_user_progress = pd.merge(
            cross_users_units, user_progress, how="outer"
        ).fillna(0)

        completed_user_progress["has_visited"] = (
            completed_user_progress["time_spent"] > 0
        ).astype(int)

        completed_user_progress["has_completed"] = (
            completed_user_progress["percentage"] >= 100
        ).astype(int)

        visited_completed_units = completed_user_progress.drop(
            columns=["percentage", "time_spent"]
        )

        visited_completed_units["visitors"] = visited_completed_units.groupby(
            grouping_vars
        )["has_visited"].transform(sum)

        visited_completed_units["finishers"] = visited_completed_units.groupby(
            grouping_vars
        )["has_completed"].transform(sum)

        visited_completed_units = pd.merge(visited_completed_units, self.visited_units)

        return visited_completed_units

    def __build_daily_effort(self):
        daily_effort = pd.merge(self.daily_effort, self.users.drop(columns="profile"))
        daily_effort["formatted_time_spent"] = pd.to_datetime(
            daily_effort["time_spent"].astype(float), unit="s"
        ).dt.strftime("%H:%M:%S")
        return daily_effort

    def __build_video_viz(self):
        video_viz_users = self.__count_viz(user_wise=True)

        video_viz_cohort = pd.concat(
            [video_viz_users, self.__count_viz(user_wise=False)]
        )

        cohort_user = (
            self.users.loc[:, ["domain", "course"]].drop_duplicates()
        )
        cohort_user["user"] = "cohort"
        cohort_user["email"] = "All students"
        cohort_user["name"] = "All students"

        users = pd.concat([self.users, cohort_user])

        video_viz_cohort = pd.merge(video_viz_cohort, users).drop(columns="profile")
        video_viz_cohort = pd.merge(video_viz_cohort, self.visited_units)
        video_viz_cohort = pd.merge(
            video_viz_cohort, self.video_current_title
        )
        video_viz_cohort = pd.merge(
            video_viz_cohort,
            self.video_current_duration.drop(columns=["duration_freq"]),
        )

        video_viz_cohort["formatted_position"] = pd.to_datetime(
            video_viz_cohort["position"].astype(float), unit="s"
        ).dt.strftime("%H:%M:%S")
        return video_viz_cohort


    def __count_viz(self, user_wise=True):
        grouping_vars = [*GROUPING_VARS, "element"]

        borders_c = self.video_borders.copy()
        borders_c.set_index(grouping_vars, inplace=True)

        borders_long = pd.DataFrame(borders_c.stack(), columns=["position"])
        borders_long.index.rename([*grouping_vars, "border_type"], inplace=True)
        borders_long.reset_index(inplace=True)
        borders_long["increment"] = 2 * (borders_long["border_type"]
                                     == "start") - 1

        if not user_wise:
            grouping_vars.remove("user")

        borders_long = borders_long.sort_values([*grouping_vars, "position"])

        borders_long["users_per_video"] = borders_long.groupby(
            grouping_vars)["user"].transform("nunique")

        borders_long["viz"] = borders_long.groupby(
            grouping_vars)["increment"].cumsum()

        borders_long.drop(columns=["border_type", "increment"], inplace=True)

        if not user_wise:
            borders_long["user"] = "cohort"

        viz = borders_long.groupby([*grouping_vars,
                               "position"]).last().reset_index()
        viz["viz"] = viz["viz"] / viz["users_per_video"]
        viz.drop(columns="users_per_video", inplace=True)

        return viz
