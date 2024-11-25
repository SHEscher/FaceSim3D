using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScreenshotTaker : MonoBehaviour
{
    public List<GameObject> Heads;

    private GameObject camera2, r_sphere2, f_sphere2;
    private Vector3 rotationAxis = new Vector3(0, 0, 1);
    private float rotationAngle = 0.0f;
    private int subdiv = 32;
    private float rotationAmount;
    private Vector3 startPosition;

    void Start()
    {
        rotationAmount = 360f / subdiv;
        camera2 = GameObject.FindGameObjectWithTag("Camera2");
        f_sphere2 = GameObject.FindGameObjectWithTag("Sphere2");
        r_sphere2 = GameObject.FindGameObjectWithTag("RotatorSphere2");
        startPosition = r_sphere2.transform.localPosition;

        foreach(GameObject head in Heads)
        {
            head.SetActive(false);
            while (head.name.Length < 3)
            {
                head.name = "0" + head.name;
            }
        }

        StartCoroutine(TakeAllScreenshots());
    }

    IEnumerator TakeAllScreenshots()
    {
        foreach(GameObject head in Heads)
        {
            rotationAngle = 0.0f;
            r_sphere2.transform.localPosition = startPosition;
            head.SetActive(true);

            // frontal screenshot
            yield return StartCoroutine(TakeOneScreenshot(head, f_sphere2, "frontal"));

            for (int i = 0; i < subdiv; i++)
            {
                // rotate sphere
                r_sphere2.transform.RotateAround(f_sphere2.transform.position, rotationAxis, -rotationAmount);
                // rotated screenshot
                yield return StartCoroutine(TakeOneScreenshot(head, r_sphere2, $"angle-{rotationAngle}"));
                rotationAngle += rotationAmount;
            }
            
            
            head.SetActive(false);
        }
    }

    IEnumerator TakeOneScreenshot(GameObject head, GameObject target, string appendName)
    {
        // rotate head
        Vector3 dir = target.transform.position - head.transform.position;
        head.transform.rotation = Quaternion.LookRotation(dir);
        // take screenshot
        yield return new WaitForEndOfFrame();
        string fileName = $"Screenshots/head-{head.name}_"+appendName+".png";
        Debug.Log(fileName);
        ScreenCapture.CaptureScreenshot(fileName);
    }
}