using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UXF;

public class TrackDB : Tracker
{
    private Dictionary<string, object> tripletsDB;
    public override string MeasurementDescriptor => "status";

    public override IEnumerable<string> CustomHeader => new string[] { "triplet", "status" };

    protected override UXFDataRow GetCurrentValues()
    {
        var values = new UXFDataRow()
        {
            ("triplet", tripletsDB.Keys),
            ("status", "U")
        };

        return values;
    }
}