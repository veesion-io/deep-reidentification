We need to make this repository useable in an online fashion. 


It will be executed lvie in the cloud on an never shutdown instance.

It should be able to receive a request containing a s3 videos directory, and a json.

The online service should : 
1. Download all data 
Specifically the directory will contain multiple folders, one for each camera from a store, and each folder contains multiple chunks of videos, as outlined in how we use those in @plan.md
2. Collate them to videos, not losing their timestamps info, jsut like in the plan.md (will downloading from s3 lose the timestamps ?)
3. Run the detection tracking and reid models.
4. Use the json : it contains the info of the track of the person we wanna track multi camera. Find the track of the person in the json, based on spatio temporal IOU with our tracks (the one in the json should be in our tracks).
5. Run the matching in a semi automatic fashion.
The matching should return a list of potential pairs it thinks the json track is associated to, in decreasing order of likelihood, for all pairs above 0.5 for reid.
6. For each pair, save a pair video (with as little computation as possible to collate them), with the query track on the left, and the candidate track on the right, witht he track visualized with bounding boxes.
7. It should send the video pairs to a tier service, TBD, and in the same go await the answer of the service (which will be human annotation of the pairs).
8. Once all pairs are annotated as being valid or not, it should build the complete video timeline of the query track, from start to finish, with as little computation as possible. If the person appears in multiple videos at the same time, it both videos should be collated in the screen without deformation as such :
1 : person appears on camera 1 from 00 to 09 -> 00 to 09 is jsut the corresponding video of camera 1
2 : they appear on camera 2 from 11 to 22 and camera 3 from 12 to 19 -> output video jumps from 09 to 11 with only camera 2, then at 12, camera 3 feed appears on the right, then at 19 it disappears with only camera 2's feed remaining.
etc... 
9. Send the video to another service for alerting the client TBD.