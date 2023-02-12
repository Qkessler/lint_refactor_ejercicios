from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict, Counter
import io
from fastavro import writer, parse_schema
from azure.storage.blob import ContainerClient
import re
import json
import os

from forma_LA.container import Container
from forma_LA.constants import VISITS_FOLDER, INTERMEDIATE_CONTAINER_NAME
from forma_LA.schemes import aggregate_schemes, create_dataframe_schema
from forma_LA.utilities import select_user, curate_video_df


class Aggregate(ABC):
    def __init__(self):
        pass

    @classmethod
    def from_container_name(cls, container_name, path):
        aggregate = cls()

        aggregate_container = Container(container_name)

        aggregate.aggregate_dict = aggregate_container.retrieve_blob(
            path, backup_container=None
        )

        aggregate.container_name = container_name

        aggregate.path = path

        aggregate_container.close()

        return aggregate

    @abstractmethod
    def as_dict(self):
        pass

    def __add_visits_dict(self):
        result = {
            "num_visits": str(self.num_visits),
            "num_visits_open": str(self.num_visits_open),
            "num_visits_lti": str(self.num_visits_lti),
            "daily_visits": self.daily_visits.astype(str).to_dict("records"),
        }
        return result

    @abstractmethod
    def upload(self, container):
        pass

    def __upload(self, container, path, schema):
        fo = io.BytesIO()
        writer(fo, parse_schema(schema), [self.as_dict()])
        ContainerClient.upload_blob(
            container.container, name=path, data=fo.getvalue(), overwrite=True
        )

    def add_path(self, intermediate):
        self.path = re.sub(r"intermediate\.avro$", "aggregate.avro", intermediate.path)

    def add_visits(self, intermediate):
        self.num_visits = intermediate.num_visits
        self.num_visits_open = intermediate.num_visits_open
        self.num_visits_lti = intermediate.num_visits_lti
        self.daily_visits = intermediate.daily_visits

    def extract_top10_units(self, intermediate):
        top10_units = intermediate.top10_units.copy()
        # while Dani uses "unit" in the panel, will change 
        top10_units["unit"] = top10_units["url"]

        self.top10_units = top10_units.sort_values(
            "num_visits", ascending=False
        ).iloc[:10, :]

        self.top10_units_lti = top10_units.sort_values(
            "num_visits_lti", ascending=False
        ).iloc[:10, :]

        self.top10_units_open = top10_units.sort_values(
            "num_visits_open", ascending=False
        ).iloc[:10, :]

        del top10_units

    def extract_top10_authors(self, intermediate):
        self.top10_authors = intermediate.top10_authors.sort_values(
            "num_visits", ascending=False
        ).iloc[:10, :]

        self.top10_authors_lti = intermediate.top10_authors.sort_values(
            "num_visits_lti", ascending=False
        ).iloc[:10, :]

        self.top10_authors_open = intermediate.top10_authors.sort_values(
            "num_visits_open", ascending=False
        ).iloc[:10, :]


    def to_json(self, path):
        folder = re.sub("aggregate.avro", "", path)
        os.makedirs(folder, exist_ok=True)
        with open(path, "a") as jsonfile:
            json.dump(self.aggregate_dict, jsonfile)
                

class AggregateRepository(Aggregate):


    def upload(self, container):
        super().upload(container)
        schema = aggregate_schemes["repository"]
        self._Aggregate__upload(container, self.path, schema)

    def as_dict(self):
        result = {
            **self._Aggregate__add_visits_dict(),
            "top10_units": self.top10_units.astype(str).to_dict("records"),
            "top10_authors": self.top10_authors.astype(str).to_dict("records"),
            "top10_units_open": self.top10_units_open.astype(str).to_dict("records"),
            "top10_authors_open": self.top10_authors_open.astype(str).to_dict(
                "records"
            ),
            "top10_units_lti": self.top10_units_lti.astype(str).to_dict("records"),
            "top10_authors_lti": self.top10_authors_lti.astype(str).to_dict("records"),
            "num_open_units": str(self.num_open_units),
            "num_open_authors": str(self.num_open_authors),
        }
        return result

    def compute_open_indicators(self):
        container = Container(INTERMEDIATE_CONTAINER_NAME)

        frequencies = Counter(
            [
                len(blob_name.split("/"))
                for blob_name in container.list_blobs()
                if blob_name.split("/")[0] == VISITS_FOLDER
            ]
        )

        container.close()

        self.num_open_units = frequencies[3]
        self.num_open_authors = frequencies[4]


class AggregateAuthor(Aggregate):

    def upload(self, container):
        super().upload(container)
        schema = aggregate_schemes["author"]
        self._Aggregate__upload(container, self.path, schema)

    def as_dict(self):
        result = {
            **self._Aggregate__add_visits_dict(),
            "top10_units": self.top10_units.astype(str).to_dict("records"),
            "top10_units_open": self.top10_units_open.astype(str).to_dict("records"),
            "top10_units_lti": self.top10_units_lti.astype(str).to_dict("records"),
        }
        return result


class AggregateUnit(Aggregate):

    def upload(self, container):
        super().upload(container)
        schema = aggregate_schemes["unit"]
        self._Aggregate__upload(container, self.path, schema)

    def as_dict(self):
        result = {
            **self._Aggregate__add_visits_dict(),
            "current_title": self.current_title,
        }
        return result


class AggregateLTI(Aggregate):

    def upload(self, container):
        super().upload(container)
        schema = self.__modify_schema_user_wide(aggregate_schemes["lti"])

        for user in [*self.users["user"].to_list(), "cohort"]:
            individual_aggregate = self.__extract_user_aggregate(user)

            path = re.sub("aggregate.avro", f"{user}/aggregate.avro", self.path)

            individual_aggregate._Aggregate__upload(container, path, schema)

    def __modify_schema_user_wide(self, schema):
        schema_names = [f["name"] for f in schema["fields"]]

        index_time_user_wide = schema_names.index("time_user_wide")
        index_percentage_user_wide = schema_names.index("percentage_user_wide")

        modified_schema = schema.copy()

        modified_schema["fields"][index_time_user_wide][
            "type"
        ] = create_dataframe_schema(self.time_user_wide.columns, "time_user_wide")

        modified_schema["fields"][index_percentage_user_wide][
            "type"
        ] = create_dataframe_schema(
            self.percentage_user_wide.columns, "percentage_user_wide"
        )

        return modified_schema

    def __extract_user_aggregate(self, user):
        user_aggregate = AggregateLTI()

        user_aggregate.number_visited_units = self.number_visited_units
        user_aggregate.number_users = self.number_users

        if not user == "cohort":
            user_aggregate.visited_completed_units = select_user(
                self.visited_completed_units, user
            )
            user_aggregate.time_user_wide = select_user(self.time_user_wide, user)
            user_aggregate.percentage_user_wide = select_user(
                self.percentage_user_wide, user
            )
            user_aggregate.daily_effort = select_user(self.daily_effort, user)
            user_aggregate.user_progress = select_user(self.user_progress, user)
            user_video_viz_cohort = curate_video_df(
                select_user(self.video_viz_cohort, user)
            )
            if len(user_video_viz_cohort) > 0:
                user_aggregate.video_viz = user_video_viz_cohort 
            user_aggregate.users = select_user(self.users, user)
        else:
            user_aggregate.visited_completed_units = (
                self.visited_completed_units.drop(
                    columns=["user", "has_visited", "has_completed"]
                ).drop_duplicates()
            )
            user_aggregate.visited_completed_units.insert(5, "user", "cohort")
            user_aggregate.visited_completed_units.insert(6, "has_visited", 1)
            user_aggregate.visited_completed_units.insert(7, "has_completed", 1)
            user_aggregate.time_user_wide = self.time_user_wide
            user_aggregate.percentage_user_wide = self.percentage_user_wide
            user_aggregate.daily_effort = self.daily_effort
            user_aggregate.user_progress = self.user_progress
            user_video_viz_cohort = curate_video_df(self.video_viz_cohort)
            if len(user_video_viz_cohort) > 0:
                user_aggregate.video_viz = user_video_viz_cohort
            user_aggregate.users = self.users

        return user_aggregate

    def as_dict(self):
        result = {
            "number_visited_units": str(self.number_visited_units),
            "number_users": str(self.number_users),
            "visited_completed_units": self.visited_completed_units.astype(str).to_dict(
                "records"
            ),
            "time_user_wide": self.time_user_wide.astype(str).to_dict("records"),
            "percentage_user_wide": self.percentage_user_wide.astype(str).to_dict(
                "records"
            ),
            "daily_effort": self.daily_effort.astype(str).to_dict("records"),
            "user_progress": self.user_progress.astype(str).to_dict("records"),
        }

        if hasattr(self, "video_viz"):
            result = {**result, "video_viz": self.video_viz}

        result = {**result, "users": self.users.astype(str).to_dict("records")}

        return result

    def build_time_percentage_user_wide(self):
        user_progress_wide = self.user_progress.pivot(
            index=["domain", "course", "user", "name", "email"],
            columns=["activity", "unit", "current_title"],
            values=["time_spent", "percentage"],
        )

        time_user_wide = user_progress_wide["time_spent"].fillna("00:00:00")
        time_user_wide = time_user_wide.droplevel([0, 1], axis=1).reset_index()

        percentage_user_wide = user_progress_wide["percentage"].fillna(0)
        percentage_user_wide = percentage_user_wide.droplevel(
            [0, 1], axis=1
        ).reset_index()

        self.time_user_wide = time_user_wide
        self.percentage_user_wide = percentage_user_wide



