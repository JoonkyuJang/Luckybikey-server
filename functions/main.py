# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`
# from firebase_functions.core import init


# ----update_feedback()------------------------------------------------------------------------------
from firebase_admin import initialize_app, firestore
from firebase_functions import https_fn, options
from copy import deepcopy


initialize_app()
firestore_client = firestore.client()


def get_connection(node1, node1_lat, node1_lon, node2, node2_lat, node2_lon):
    if node1 != node2:
        return node2
    else:
        tile = f"seoul_tile_lat_{int(node1_lat*100)}_lng_{int(node1_lon*100)}"
        connections = firestore_client.collection("node_tiles").document(tile).collection("nodes").document(node1).get().to_dict()["connections"]
        for id, value in connections.items():
            # print(f"id: {id}, lat: {value['lat']}, lon: {value['lon']}")
            for point in value["routes"][0]["branch"]:
                if point["lat"] == node2_lat and point["lon"] == node2_lon:
                    return id
        raise ValueError("Connection not found")


@https_fn.on_call(memory=options.MemoryOption.GB_1, region="asia-northeast3")
def update_feedback(req):
    """
    connection: {node1: str, node2: str}
    label: str
    pref: {'i': int} for i in range(8)
    """
    pass
    try:  # 요청 데이터 파싱
        connection = req.data["connection"]
        node1 = str(connection["node1"])
        node1_lat = float(connection["node1_lat"])
        node1_lon = float(connection["node1_lon"])
        node2 = str(connection["node2"])
        node2_lat = float(connection["node2_lat"])
        node2_lon = float(connection["node2_lon"])
        label = req.data["label"]
        pref = req.data["pref"]
    except KeyError as e:
        print(f"KeyError: {e}")
        return {"error": "Invalid request data."}

    node2 = get_connection(node1, node1_lat, node1_lon, node2, node2_lat, node2_lon)

    # node1 문서에 적힌 node2와의 connection 정보를 먼저 갱신
    tile1 = f"seoul_tile_lat_{int(node1_lat*100)}_lng_{int(node1_lon*100)}"
    doc_ref = firestore_client.collection("node_tiles").document(tile1).collection("nodes").document(node1)
    feedback_data = doc_ref.get().to_dict()["connections"][f"{node2}"]["clusters"][f"{label}"]["feedback"]
    attributes_data = doc_ref.get().to_dict()["connections"][f"{node2}"]["clusters"][f"{label}"]["attributes"]
    new_attributes = deepcopy(attributes_data)

    print(f"label: {label}")
    print(f"pref: {pref}")
    print(f"feedback data: {feedback_data}")

    for i in range(8):
        if pref[i] == 1:
            feedback_data[i] += 1
            if feedback_data[i] == 30:
                feedback_data[i] = 0
                new_attributes[i] = min(1, new_attributes[i] + 0.01)
        if pref[i] == -1:
            feedback_data[i] -= 1
            if feedback_data[i] == -30:
                feedback_data[i] = 0
                new_attributes[i] = max(0, new_attributes[i] - 0.01)

    print(f"feedback data: {feedback_data}")

    doc_ref.update({f"connections.{node2}.clusters.{label}.feedback": feedback_data})

    tile2 = f"seoul_tile_lat_{int(node2_lat*100)}_lng_{int(node2_lon*100)}"
    firestore_client.collection("node_tiles").document(tile2).collection("nodes").document(node2).update({f"connections.{node1}.clusters.{label}.feedback": feedback_data})
    if new_attributes != attributes_data:
        doc_ref.update({f"connections.{node2}.clusters.{label}.attributes": new_attributes})
        firestore_client.collection("node_tiles").document(tile2).collection("nodes").document(node2).update({f"connections.{node1}.clusters.{label}.attributes": new_attributes})

    print(f"node1: {node1}")
    print(f"tile1: {tile1}")
    print(f"node2: {node2}")
    print(f"tile2: {tile2}")


# ---------------------------------------------------------------------------------------------------

# ----get_rankings()---------------------------------------------------------------------------------
from firebase_admin import initialize_app, firestore
from firebase_functions import https_fn, options


# initialize_app()
# firestore_client = firestore.client()


@https_fn.on_call(memory=options.MemoryOption.GB_1, region="asia-northeast3")
def get_rankings(req):
    try:  # 요청 데이터 파싱
        uid = str(req.data["uid"])
    except Exception as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=(f"Missing argument '{e}' in request data. {repr(e)}"),
        )

    user_ref = firestore_client.collection("users").document(uid)
    user_data = user_ref.get().to_dict()
    all_users = firestore_client.collection("users").order_by("totalDistance", direction=firestore.Query.DESCENDING).stream()
    ranking = -1
    top_10_users = []
    for i, user in enumerate(all_users):
        if i < 10:
            user_dict = user.to_dict()
            if "fullname" not in user_dict:
                firestore_client.collection("users").document(user.id).update({"fullname": "Unknown"})
                user_dict["fullname"] = "Unknown"
            top_10_users.append(
                {
                    "fullname": user_dict["fullname"],
                    "totalDistance": user_dict["totalDistance"],
                }
            )
        elif ranking != -1:
            break
        if user.id == uid:
            ranking = i + 1
            if i >= 10:
                break

    return {
        "fullname": user_data["fullname"],
        "ranking": ranking,
        "totalDistance": user_data["totalDistance"],
        "top_10_users": top_10_users,
    }


# ---------------------------------------------------------------------------------------------------


# ----generate_custom_token()------------------------------------------------------------------------
from firebase_admin import initialize_app, firestore, auth
from firebase_functions import https_fn, options


# initialize_app()
# firestore_client = firestore.client()


@https_fn.on_call(memory=options.MemoryOption.GB_1, region="asia-northeast3")
def generate_custom_token(request: https_fn.CallableRequest):
    try:
        user_id = request.data["token"]
    except Exception as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=f"Missing argument 'user_id' in request data. {e}",
        )

    try:
        token = auth.create_custom_token(user_id).decode("utf-8")
    except Exception as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"Failed to create custom token. {e}",
        )

    return {"token": token}


# ---------------------------------------------------------------------------------------------------

# ----cluster_users()--------------------------------------------------------------------------------
from pandas import DataFrame
from sklearn.cluster import KMeans
from zoneinfo import ZoneInfo

from firebase_admin import initialize_app, firestore
from firebase_functions import scheduler_fn, options


# initialize_app()
# firestore_client = firestore.client()


class User:
    def __init__(self, uid: str, attributes: dict):
        self.uid = uid
        self.attributes = [value for _, value in attributes.items()]


# 클러스터링 진행하는 코드
@scheduler_fn.on_schedule(
    schedule="0 0 * * 0",
    timezone=ZoneInfo("Asia/Seoul"),
    memory=options.MemoryOption.GB_1,
    region="asia-northeast3",
)
def cluster_users(event: scheduler_fn.ScheduledEvent):
    users_ref = firestore_client.collection("users")

    docs = users_ref.select(["attributes"]).stream()  # docs는 제너레이터 객체. 좀 큰 데이터에서 받아올 때는 stream으로
    docs1 = firestore_client.collection("clusters").get()  # docs1은 리스트 객체. 작은 데이터라 get으로

    Users = [User(doc.id, doc.to_dict()["attributes"]) for doc in docs]
    Datas = DataFrame.from_dict({user.uid: user.attributes for user in Users}, orient="index")
    num_users = len(Users)

    # 유저가 하나면 클러스터링을 진행하는 것이 아니라 그냥 그걸로 끝내기
    if num_users == 0:
        exit()

    if num_users == 1:
        users_ref.document(Users[0].uid).update({"label": 0})

        centroid = list(Users[0].attributes)
        data = {
            "centroid": centroid,
        }
        firestore_client.collection("clusters").document(str(0)).set(data)
    elif not docs1:  # 이전에 클러스터링을 한 적이 없으면
        # N = 16으로 클러스터링
        kmeans = KMeans(n_clusters=16, init="k-means++", random_state=0).fit(Datas)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        for i, user in enumerate(Users):
            users_ref.document(user.uid).update({"label": int(labels[i])})
        for i, centroid in enumerate(centroids):
            data = {
                "centroid": list(centroid),
            }
            firestore_client.collection("clusters").document(str(i)).set(data)
    else:  # 이전에 클러스터링을 한 적이 있으면 그냥 클러스터링 다시 돌리기
        docs1_sorted = sorted(docs1, key=lambda x: int(x.id))
        centroids = [doc.to_dict()["centroid"] for doc in docs1_sorted]  # 이전의 중심점들 정보를 받아오기

        kmeans = KMeans(n_clusters=len(centroids), init=centroids, random_state=0).fit(Datas)
        labels0 = kmeans.labels_
        centroids0 = list(kmeans.cluster_centers_)

        for i, user in enumerate(Users):
            users_ref.document(user.uid).update({"label": int(labels0[i])})
        for i, centroid in enumerate(centroids0):
            data = {
                "centroid": list(centroid),
            }
            print(f"document {i} is updated to {data}")
            firestore_client.collection("clusters").document(str(i)).set(data)


# ---------------------------------------------------------------------------------------------------

# ----assign_label()---------------------------------------------------------------------------------
from firebase_functions.firestore_fn import on_document_updated
from firebase_admin import initialize_app, firestore
from firebase_functions import options


# initialize_app()
# firestore_client = firestore.client()


@on_document_updated(
    document="users/{user_id}",
    memory=options.MemoryOption.GB_1,
    region="asia-northeast3",
)
def assign_label(event) -> None:
    new_value = event.data.after
    prev_value = event.data.before
    if new_value.get("attributes") == prev_value.get("attributes"):
        return

    user_id = new_value.get("uid")
    attributes_dict = new_value.get("attributes")
    keys = (
        "scenery",
        "safety",
        "traffic",
        "fast",
        "signal",
        "uphill",
        "bigRoad",
        "bikePath",
    )
    attributes = [attributes_dict[key] for key in keys]
    # 각 중심점과 거리 비교하여 가장 가까운 중심점의 라벨을 할당
    clusters_ref = firestore_client.collection("clusters")
    docs = clusters_ref.get()
    min_dist = float("inf")
    label = -1
    for doc in docs:
        dist = sum((a - b) ** 2 for a, b in zip(attributes, doc.to_dict()["centroid"]))
        if dist < min_dist:
            min_dist = dist
            label = doc.id

    firestore_client.collection("users").document(user_id).update({"label": int(label)})


# ---------------------------------------------------------------------------------------------------

# ----request_route()-------------------------------------------------------------------------
from firebase_admin import initialize_app, firestore, credentials
from firebase_functions import https_fn, options
from google.cloud.firestore import CollectionReference

from typing import Dict, List, TypedDict, Set, Tuple
from geopy.distance import distance
import heapq
from numpy import array, dot

import time
from datetime import datetime
import asyncio
import multiprocessing
from multiprocessing import Queue, Process

AStarReturn = TypedDict("AStarReturn", {"path": List[Dict[str, float]], "full_distance": float})
RequestRouteReturn = TypedDict(
    "RequestRouteReturn",
    {
        "route": List[Dict[str, float]],
        "path": List[Dict[str, float]],
        "full_distance": float,
    },
)

# initialize_app()
# firestore_client = firestore.client()

TILES_COLLECTION = "node_tiles"
NODES_COLLECTION = "nodes"
NUM_PROCESSES = 8


class Node:
    def __init__(self, id: int, geometry: Dict[str, float], connections, parent=None):
        self.id: int = id
        self.lat: float = geometry["lat"]
        self.lon: float = geometry["lon"]
        self.connections = connections
        self.g: float = 0
        self.h: float = 90000
        self.f: float = 0
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.id == other.id


async def firestore_get(tile_input_queue: Queue, node_output_queue: Queue):
    try:
        collection_ref = firestore_client.collection(TILES_COLLECTION)
        while True:
            tile = tile_input_queue.get()
            if tile is None:
                break
            if tile == "BUFFER":
                continue
            print(f"Process {multiprocessing.current_process().name} started pushing nodes into the queue")
            for doc in collection_ref.document(tile).collection(NODES_COLLECTION).stream():
                node = Node(
                    int(doc.id),
                    doc.to_dict(),
                    {int(k): v for k, v in doc.to_dict()["connections"].items()},
                )
                node_output_queue.put(node)
                # print(f"Process {multiprocessing.current_process().name} pushed node {doc.id} into the queue")
            print(f"{multiprocessing.current_process().name} finished pushing {tile} nodes into the queue")
    except Exception as e:
        print(f"Error in process {multiprocessing.current_process().name}: {e}")
    finally:
        print(f"{multiprocessing.current_process().name} exiting")
        tile_input_queue.close()


async def node_put(node_queue, node_map):
    try:
        while True:
            data = node_queue.get()
            if data is None:
                break
            node_map[data.id] = data
            # print(f"Main process got node {data.id} from the queue")
    except Exception as e:
        print(f"Error in main process: {e}")
    finally:
        print("Main process exiting")
        node_queue.close()


def getter(tile_input_queue, node_output_queue):
    # run the get function in the child process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(firestore_get(tile_input_queue, node_output_queue))
    loop.close()


def putter(node_queue: Queue, node_map: Dict[int, Node]):
    # run the put function in the main process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(node_put(node_queue, node_map))
    loop.close()


def create_node_map(
    node_map: Dict[int, Node],
    open_tiles: Set,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
):
    first_tile, last_tile, tiles = get_tiles(start_lat, start_lon, end_lat, end_lon, open_tiles)

    getter_processes = []
    node_output_queue = Queue()

    for _ in range(NUM_PROCESSES):
        tile_input_queue = Queue()
        p = Process(target=getter, args=(tile_input_queue, node_output_queue))
        getter_processes.append({"process": p, "tile_input_queue": tile_input_queue})
        p.start()
        print(f"Process {p.name} started")

    putter_process = Process(target=putter, args=(node_output_queue, node_map))
    putter_process.start()
    # print(f"Process {putter_process.name} started")

    # todo: optimize order of tiles to fetch based on the start and end points
    for i, tile in enumerate(tiles):
        getter_processes[i % (NUM_PROCESSES - 1) + 1]["tile_input_queue"].put(tile)

    # Prioritize the first and last tiles to be fetched
    # to ensure that the start and end points are covered
    collection_ref = firestore_client.collection(TILES_COLLECTION)
    for doc in collection_ref.document(first_tile).collection(NODES_COLLECTION).stream():
        node = Node(
            int(doc.id),
            doc.to_dict(),
            {int(k): v for k, v in doc.to_dict()["connections"].items()},
        )
        node_output_queue.put(node)

    for doc in collection_ref.document(last_tile).collection(NODES_COLLECTION).stream():
        node = Node(
            int(doc.id),
            doc.to_dict(),
            {int(k): v for k, v in doc.to_dict()["connections"].items()},
        )
        node_output_queue.put(node)

    time.sleep(0.5)

    return getter_processes, {
        "process": putter_process,
        "node_output_queue": node_output_queue,
    }


def get_routes(
    node_map,
    start_node: Node,
    end_node: Node,
    user_taste: bool,
    user_group: str,
    group_preference: List,
    open_tiles: Set,
    priority_queue,
) -> List[AStarReturn]:

    routes = []
    # 0 취향 추천
    routes.append(
        astar_road_finder2(
            node_map,
            start_node,
            end_node,
            user_taste,
            user_group,
            group_preference,
            open_tiles,
            priority_queue,
        )
    )

    # 1 Fatest route
    routes.append(
        astar_road_finder2(
            node_map,
            start_node,
            end_node,
            False,
            user_group,
            group_preference,
            open_tiles,
            priority_queue,
        )
    )

    # 2 풍경좋은 경로

    routes.append(
        astar_road_finder2(
            node_map,
            start_node,
            end_node,
            True,
            user_group,
            [1, 0, 0, 0, 0, 0, 0, 0],
            open_tiles,
            priority_queue,
        )
    )

    # 3 큰 길

    routes.append(
        astar_road_finder2(
            node_map,
            start_node,
            end_node,
            True,
            user_group,
            [0, 1, 0, 0, 0, 0, 1, 0],
            open_tiles,
            priority_queue,
        )
    )

    # 4 자전거 길

    routes.append(
        astar_road_finder2(
            node_map,
            start_node,
            end_node,
            True,
            user_group,
            [0, 0, 0, 0, 0, 0, 0, 1],
            open_tiles,
            priority_queue,
        )
    )

    return routes


def get_tiles(start_lat: float, start_lon: float, end_lat: float, end_lon: float, open_tiles: Set) -> Tuple[str, str, List[str]]:
    smallest_lat = min(start_lat, end_lat)
    smallest_lon = min(start_lon, end_lon)
    largest_lat = max(start_lat, end_lat)
    largest_lon = max(start_lon, end_lon)

    first_tile = f"seoul_tile_lat_{int(start_lat * 100)}_lng_{int(start_lon * 100)}"
    last_tile = f"seoul_tile_lat_{int(end_lat * 100)}_lng_{int(end_lon * 100)}"
    open_tiles.add(first_tile)
    open_tiles.add(last_tile)

    tiles = []
    for lat in range(int(smallest_lat * 100), int(largest_lat * 100) + 1):
        for lon in range(int(smallest_lon * 100), int(largest_lon * 100) + 1):
            if lat == int(start_lat * 100) and lon == int(start_lon * 100):
                continue
            if lat == int(end_lat * 100) and lon == int(end_lon * 100):
                continue
            tile = f"seoul_tile_lat_{lat}_lng_{lon}"
            tiles.append(tile)
            open_tiles.add(tile)

    return first_tile, last_tile, tiles


def get_node(node_map, id: int, lat: float, lon: float, open_tiles: Set, priority_queue) -> Node:
    tile = f"seoul_tile_lat_{int(lat * 100)}_lng_{int(lon * 100)}"
    # Firestore에서 노드 생성
    if id in node_map:
        return node_map[id]
    elif tile in open_tiles:
        print("get new node from node map")
        start_time = datetime.now()
        while True:
            if id in node_map:
                print("find new node from node map")
                print(id)
                return node_map[id]
            time.sleep(0.1)
    else:
        # tile이 open_tiles에 없으면 getter process에 요청
        print("get node from new tile")
        priority_queue.put(tile)
        open_tiles.add(tile)
        start_time = datetime.now()
        while True:
            if id in node_map:
                print("done")
                return node_map[id]
            time.sleep(0.1)


def get_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # 두 지점 사이의 거리 계산
    coords1 = (lat1, lon1)
    coords2 = (lat2, lon2)
    return round(distance(coords1, coords2).m, 3)


def heuristic_Manhattan_distance(cur_node: Node, end_node: Node) -> float:

    # 거리 계산
    h = distance((cur_node.lat, cur_node.lon), (end_node.lat, end_node.lon)).meters
    return h


def heuristic_preference_distance(cur_node: Node, end_node: Node, group_road_type, group_preference) -> float:
    manhattan_dist = heuristic_Manhattan_distance(cur_node, end_node)
    # next node의 해당 group의 preference 추가
    # print(group_road_type)

    feature_num = len(group_preference)  # feature_num을 고정 값인 preference를 다 더한 값을 사용하면 어떻게 될까.. 음수가 될 수도 있긴한데 음수를 0으로 빼버리면?
    pref_sum = abs(sum(group_preference))
    lt = array(group_road_type)
    gp = array(group_preference)
    road_preference = dot(lt, gp)

    if all(abs(x) < 0.3 for x in group_preference):  # group의 preference이기 때문에 이미 0.3보다 작은 애들은 preference를 끄고 진행한다고 생각하고 코드 짜기
        pref_sum = feature_num

    # feature 개수로 나눈 대로 scaling
    pref_dist = manhattan_dist - (manhattan_dist / pref_sum) * road_preference
    if pref_dist < 0:
        pref_dist = 0  # 휴리스틱이 항상 0 이상이도록

    return pref_dist


def astar_road_finder2(
    node_map,
    start_node: Node,
    end_node: Node,
    user_taste: bool,
    user_group: str,
    group_preference: List,
    open_tiles: Set,
    priority_queue,
) -> AStarReturn:
    # A* 알고리즘을 사용하여 시작 노드에서 도착 노드까지의 최단 경로 찾기
    open_list: List[Node] = []
    closed_set = set()
    start_node.g = 0
    start_node.h = heuristic_Manhattan_distance(start_node, end_node)
    heapq.heappush(open_list, start_node)

    while open_list != []:
        cur_node = heapq.heappop(open_list)
        closed_set.add(cur_node.id)

        if cur_node == end_node:
            final_road = []
            final_path = [{"node_id": cur_node.id, "lat": cur_node.lat, "lon": cur_node.lon}]
            total_distance = cur_node.g
            while cur_node is not None:
                final_road.append({"node_id": cur_node.id, "lat": cur_node.lat, "lon": cur_node.lon})
                if cur_node.parent is not None:
                    final_path += [
                        {
                            "node_id": cur_node.id,
                            "lat": branch["lat"],
                            "lon": branch["lon"],
                        }
                        for branch in cur_node.connections[cur_node.parent.id]["routes"][0]["branch"][1:]
                    ]
                cur_node = cur_node.parent
            return {
                "route": final_road[::-1],
                "path": final_path[::-1],
                "full_distance": total_distance,
            }

        for id, inner_dict in cur_node.connections.items():
            new_node = get_node(
                node_map,
                int(id),
                inner_dict["lat"],
                inner_dict["lon"],
                open_tiles,
                priority_queue,
            )

            if new_node.id in closed_set:
                continue
            if new_node in open_list:
                if (cur_node.g + inner_dict["distance"]) >= new_node.g:
                    continue
            new_node.g = cur_node.g + inner_dict["distance"]
            if user_taste:
                new_node.h = heuristic_preference_distance(
                    new_node,
                    end_node,
                    inner_dict["clusters"][user_group]["attributes"],
                    group_preference,
                )
            else:
                new_node.h = heuristic_Manhattan_distance(new_node, end_node)
            new_node.f = new_node.g + new_node.h
            new_node.parent = cur_node
            heapq.heappush(open_list, new_node)

    # 길이 연결되지 않았으면 에러 발생
    raise https_fn.HttpsError(
        code=https_fn.FunctionsErrorCode.INTERNAL,
        message="No route was found between the start and end points.",
    )


def get_nearest_node2(node_map, lat: float, lon: float) -> tuple[int, float]:
    # 기준 좌표 부근에서 후보 노드들 query
    docs = []

    for node in node_map.values():
        if lat - 0.005 <= node.lat <= lat + 0.005 and lon - 0.005 < node.lon < lon + 0.005:
            docs.append(node)

    # 해당 범위에 노드가 없으면 에러 발생
    if not docs:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message="No nodes near the end point were found.",
        )

    # 후보 노드들 중 가장 가까운 노드 찾기
    min = float("inf")
    node_min: Node = None
    for node in docs:
        dist = get_distance(node.lat, node.lon, lat, lon)
        if dist < min:
            node_min = node
            min = dist

    return (node_min, min)


def exit_processes(getter_processes, putter_process):
    for p in getter_processes:
        p["tile_input_queue"].put(None)
        p["tile_input_queue"].close()

    for p in getter_processes:
        p["process"].join()

    putter_process["node_output_queue"].put(None)
    putter_process["node_output_queue"].close()
    putter_process["process"].join()


@https_fn.on_call(timeout_sec=600, memory=options.MemoryOption.GB_4, region="asia-northeast3")
def request_route(req: https_fn.CallableRequest) -> RequestRouteReturn:
    try:  # 요청 데이터 파싱
        start_point = req.data["StartPoint"]
        end_point = req.data["EndPoint"]
        user_taste = req.data["UserTaste"]
        user_group = req.data["UserGroup"]
        group_preference = req.data["GroupPreference"]
    except KeyError as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=(f"Missing argument '{e}' in request data."),
        )

    try:  # 요청 데이터 유효성 검사
        start_lat = start_point["lat"]
        start_lon = start_point["lon"]
    except KeyError as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=(f"Missing argument '{e}' in start point."),
        )

    try:  # 요청 데이터 유효성 검사
        end_lat = end_point["lat"]
        end_lon = end_point["lon"]
    except KeyError as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=(f"Missing argument '{e}' in end point."),
        )

    try:  # 요청 데이터 타입 변환
        start_lat = float(start_lat)
        start_lon = float(start_lon)
        end_lat = float(end_lat)
        end_lon = float(end_lon)
        user_taste = bool(user_taste)
        user_group = str(user_group)
        group_preference = list(group_preference)
    except ValueError as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=(e.args[0]),
        )

    try:
        # Spawn subprocesses to fetch nodes from Firestore
        # and spawn a process to put nodes into the node_map
        # dynamically change the priority of tiles to fetch based on the start and end points
        # change the number of processes to fetch nodes from Firestore
        print("Creating node map...")
        node_map = multiprocessing.Manager().dict()
        open_tiles = set()
        getter_processes, putter_process = create_node_map(node_map, open_tiles, start_lat, start_lon, end_lat, end_lon)
        priority_queue = getter_processes[0]["tile_input_queue"]
        print("Node map created.")
        print(len(node_map))

    except Exception as e:
        exit_processes(getter_processes, putter_process)
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"Failed to create node map. Error: {e.args[0]}",
        )

    try:  # 시작점에서 가장 가까운 노드 찾기
        nearest_start_node, start_dist = get_nearest_node2(node_map, start_lat, start_lon)
        print(f"Nearest start node: {nearest_start_node.id}, distance: {start_dist}")
    except Exception as e:
        exit_processes(getter_processes, putter_process)
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"No nodes near the start point were found. Error: {e.args}",
        )

    try:  # 도착점에서 가장 가까운 노드 찾기
        nearest_end_node, end_dist = get_nearest_node2(node_map, end_lat, end_lon)
        print(f"Nearest end node: {nearest_end_node.id}, distance: {end_dist}")
    except Exception as e:
        exit_processes(getter_processes, putter_process)
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"No nodes near the end point were found. Error: {e.args}",
        )

    try:  # 시작노드-도착노드 길찾기
        print("Finding route...")
        results = get_routes(
            node_map,
            start_node=nearest_start_node,
            end_node=nearest_end_node,
            user_taste=user_taste,
            user_group=user_group,
            group_preference=group_preference,
            open_tiles=open_tiles,
            priority_queue=priority_queue,
        )
        exit_processes(getter_processes, putter_process)
    except Exception as e:
        exit_processes(getter_processes, putter_process)
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"An error occured while running a star. Error: {repr(e)}",
        )

    # 시작점과 도착점을 최종 경로에 추가
    try:
        start_point_node = [{"node_id": None, "lat": start_lat, "lon": start_lon}]
        end_point_node = [{"node_id": None, "lat": end_lat, "lon": end_lon}]

        routes = []
        for i in range(len(results)):
            routes.append(
                {
                    "path": start_point_node + results[i]["path"] + end_point_node,
                    "full_distance": start_dist + results[i]["full_distance"] + end_dist,
                }
            )

        print("Returning routes")
        return routes
    except Exception as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=(e.args[0]),
        )


# ---------------------------------------------------------------------------------------------------
