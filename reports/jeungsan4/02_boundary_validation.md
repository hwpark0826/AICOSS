# 02. 경계·공간 단위 검증

대상 코드 `3110451`는 원본 Shapefile에서 **1개 피처**로 확인되었습니다. DBF 속성의 상권명·유형·자치구·행정동은 폴리곤 피처와 동일한 코드에서 읽었습니다. 좌표계는 `.prj`에 기록된 Korea_2000_Korea_Central_Belt이며 단위는 m입니다.

| target_area_code | target_area_name | polygon_feature_count_for_code | matching_code_or_name_feature_count | polygon_area_m2_calculated | dbf_reported_area_m2 | area_difference_m2 | perimeter_m | polygon_centroid_x | polygon_centroid_y | dbf_centroid_x | dbf_centroid_y | centroid_difference_m | coordinate_reference | spatial_snapshot_count | boundary_change_verifiable | store_point_data_available | nearby_selected_area_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3110451 | 증산역 4번 | 1 | 1 | 178167.0054397583 | 178167.0 | 0.00543975830078125 | 2755.094677761576 | 192056.20773387764 | 454140.83375653805 | 192207.0 | 454149.0 | 151.01322807819 | Korea_2000_Korea_Central_Belt (metre units; source .prj) | 1 | False | False | 19 |

현재 프로젝트에는 한 시점의 폴리곤만 있으므로 **경계 변경 여부는 검증 불가**입니다. 점포 좌표 파일도 없어 점포가 경계 밖으로 이동했는지는 판단하지 않았습니다. 지도는 `outputs/jeungsan4/figures/jeungsan4_boundary_map.png`에 저장했습니다.
