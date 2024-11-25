using UnityEngine;
using System.Globalization;
using System.Threading;

public class CultureSetter : MonoBehaviour
{
    private void Awake()
    {
        Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
    }
}