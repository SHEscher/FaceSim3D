using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRotator : MonoBehaviour
{
    /*
    // Camera real-time rotation controllers
    public float time;
    public float initAngle;
    public float finalAngle;
    private float angleMove;
    private float angleCovered;
    private float speed;
    private bool angularDirection;
    private int nDirChanges;
    // Start is called before the first frame update
    private void Start()
    {
        transform.Rotate(0, initAngle, 0);
        angleCovered = 0;
        speed = (finalAngle - initAngle) / time;
        angleMove = speed * Time.deltaTime;
        angularDirection = false;
        nDirChanges = 1;
    }
    // Update is called once per frame
    void Update()
    {
        // Debug.Log(transform.rotation.y);
        // Camera real-time rotation controllers        
        if(angularDirection)
        {
            if (angleCovered > nDirChanges*(finalAngle - initAngle))
            {
                transform.Rotate(0, -angleMove, -angleMove);
                angleCovered = angleCovered + angleMove;
            }
            else
            {
                angularDirection = false;
                nDirChanges += 1;
                // Debug.Log("Direction change: Left");
            }

        }
        else
        {
            if (angleCovered > nDirChanges*(finalAngle - initAngle))
            {
                transform.Rotate(0, angleMove, angleMove);
                angleCovered = angleCovered + angleMove;
            }
            else
            {
                angularDirection = true;
                nDirChanges += 1;
                // Debug.Log("Direction change: Right");
            }
        }
    }
    */
}
