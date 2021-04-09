# Motion matching

- basics + mocap + optimization + animation displacement: https://www.gdcvault.com/play/1023280/Motion-Matching-and-The-Road
- Lot of trajectories: https://www.youtube.com/watch?v=z_wpgHFSWss
- https://www.youtube.com/watch?v=KSTn3ePDt50&feature=youtu.be

```
 code of motion matching  Collapse source
int m_CurrentAnimIndex;
float m_CurrentAnimTime;
void AmoUpdate(Goal goal, float dt)
{
  m_CurrentAnimTime +=dt;
  Pose currentPose = EvaluateLerpedPoseFromData(m_currentAnimIndex, m_CurrentAnimTime); //interpolate animation
  float bestCost = 10000.f;
  Pose bestPose;
  //loop on all mocap data
  for (int i=0; i<m_Poses.Size(); i++)
  {
    Pose candidatePose  = m_Poses[i];
    float thisCost = ComputeCost(currentPose, candidatePose, goal);
    if ( thisCost < bestCost )
    {
      bestCost = thisCost;
      bestPose = CandidatePose;
    }
  }
}
//Blend each frame!

float ComputeCost(Pose currentPose, Pose candidatePose, Goal goal)
{
  float cost = 0.0f;
  cost += ComputeCost(currentPose, candidatePose);
  cost += ComputeFutureCost(candidatePose, goal);
  return cost;
}

class TrajectoryPoint
{
  Vector3 m_Position;
  float m_Sight;
  float m_TimeDelay;
};

class Goal
{
  Array<TrajectroyPoint> m_DesiredTrajectory;
  Stance m_DesiredStance;
  // ...'Goal' is at least local velocity, feet positions, feet velocites and position of weapon
 // And 'tag' of course
};
//Tips:
//Ground feet with IK it they were grounded in animation
//Rotate entity and animation to fit desired trajectory and orientation
//Timescale animations if needed up to 30%
//serialize events and control data to debug
//Use LODs and KD-Tree for optimizations
//Never bring leg longer than it was in animation
```

- https://rockhamstercode.tumblr.com/post/178388643253/motion-matching
- https://rockhamstercode.tumblr.com/post/175020832938/predicting-is-guesswork
- https://www.gdcvault.com/play/1023316/Fitting-the-World-A-Biomechanical
  - peeks - rotate around grounded foot
  - limit IK \procedural rotations and movements of
  - keep knees direction

# Classics

- Overgrowth: https://www.youtube.com/watch?v=LNidsMesxSE

# Physics based

- https://www.gdcvault.com/search.php#&conference_id=&category=free&firstfocus=&keyword=uncharted+animation
- https://www.gdcvault.com/search.php#&conference_id=&category=free&firstfocus=&keyword=ea+physics
- https://www.gdcvault.com/play/1024087/Physics-Animation-in-Uncharted-4
- https://www.gdcvault.com/play/1025210/Physics-Driven-Ragdolls-and-Animation

# IK rig

- https://www.youtube.com/watch?v=SQo9pTQ14Jk (https://github.com/sketchpunk/fungi)
- Ubisoft: https://www.youtube.com/watch?v=KLjTU0yKS00

# Math

- IK: https://www.gdcvault.com/play/1022147/Math-for-Game-Programmers-Inverse

# Assimp

- http://www.ogldev.org/www/tutorial38/tutorial38.html - skeletal animations

# Vulkan

- Synchronization https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
- Validate synchronization https://www.lunarg.com/wp-content/uploads/2021/01/Final_Guide-to-Vulkan-Synchronization-Validation_Jan_21.pdf
- Render passes https://vulkan.lunarg.com/doc/view/1.2.170.0/linux/tutorial/html/10-init_render_pass.html
- Push constants https://vkguide.dev/docs/chapter-3/push_constants/
  - how to bind several ranges: https://stackoverflow.com/questions/37056159/using-different-push-constants-in-different-shader-stages

# Repeating patters

- Quasi crystals and aperiodic tilings https://www.youtube.com/watch?v=48sCx-wBs34
