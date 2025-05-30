package mybots;

import robocode.*;
import robocode.util.Utils;

import java.io.*;
import java.util.*;

public class DataCollectingBot extends AdvancedRobot {
    private PrintWriter writer;
    private Map<Bullet, BulletData> bulletMap = new HashMap<>();

    public void run() {
        try {
            writer = new PrintWriter(new BufferedWriter(new FileWriter(getDataFile("battle_data.csv"), true)));
            writer.println("bot_x,bot_y,gun_heading,enemy_x,enemy_y,distance,enemy_heading,enemy_velocity,enemy_angle,fire_power,bullet_speed,hit");
        } catch (IOException e) {
            e.printStackTrace();
        }

        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);

        while (true) {
            turnRadarRight(360); // Skanuj przeciwnika
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        try {
            double myX = getX();
            double myY = getY();
            double myGunHeading = getGunHeadingRadians();

            double enemyDistance = e.getDistance();
            double enemyBearing = e.getBearingRadians();
            double absoluteBearing = getHeadingRadians() + enemyBearing;

            double enemyX = myX + enemyDistance * Math.sin(absoluteBearing);
            double enemyY = myY + enemyDistance * Math.cos(absoluteBearing);
            double enemyHeading = e.getHeadingRadians();
            double enemyVelocity = e.getVelocity();
            double angleToEnemy = Utils.normalRelativeAngle(absoluteBearing - myGunHeading);

            double firePower = 2.0;
            double bulletSpeed = 20 - 3 * firePower;

            setTurnGunRightRadians(angleToEnemy);
            if (getGunHeat() == 0) {
                Bullet bullet = fireBullet(firePower);
                if (bullet != null) {
                    bulletMap.put(bullet, new BulletData(myX, myY, myGunHeading, enemyX, enemyY, enemyDistance,
                            enemyHeading, enemyVelocity, angleToEnemy, firePower, bulletSpeed));
                    System.out.println("Oddano strzał!");
                } else {
                    System.out.println("Nie udało się oddać strzału (bullet == null)");
                }
            }

            // Poruszaj się w stronę przeciwnika
            double targetAngle = absoluteBearing - getHeadingRadians();
            setTurnRightRadians(Utils.normalRelativeAngle(targetAngle));
            setAhead(enemyDistance - 100); // Zatrzymaj się ~100 jednostek od przeciwnika

            execute();

        } catch (Exception ex) {
            System.out.println("Błąd w onScannedRobot: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    public void onBulletHit(BulletHitEvent e) {
        Bullet b = e.getBullet();
        BulletData data = bulletMap.get(b);
        if (data != null) {
            data.hit = 1;
            writer.println(data.toCsv());
            writer.flush();
            bulletMap.remove(b);
            System.out.println("Trafiono przeciwnika!");
        }
    }

    public void onBulletMissed(BulletMissedEvent e) {
        Bullet b = e.getBullet();
        BulletData data = bulletMap.get(b);
        if (data != null) {
            data.hit = 0;
            writer.println(data.toCsv());
            writer.flush();
            bulletMap.remove(b);
            System.out.println("Pudło!");
        }
    }

    public void onDeath(DeathEvent e) {
        if (writer != null) {
            writer.close();
        }
    }

    static class BulletData {
        double botX, botY, gunHeading;
        double enemyX, enemyY, distance, enemyHeading, enemyVelocity, enemyAngle;
        double firePower, bulletSpeed;
        int hit;

        public BulletData(double botX, double botY, double gunHeading, double enemyX, double enemyY, double distance,
                          double enemyHeading, double enemyVelocity, double enemyAngle,
                          double firePower, double bulletSpeed) {
            this.botX = botX;
            this.botY = botY;
            this.gunHeading = gunHeading;
            this.enemyX = enemyX;
            this.enemyY = enemyY;
            this.distance = distance;
            this.enemyHeading = enemyHeading;
            this.enemyVelocity = enemyVelocity;
            this.enemyAngle = enemyAngle;
            this.firePower = firePower;
            this.bulletSpeed = bulletSpeed;
            this.hit = -1;
        }

        public String toCsv() {
            return String.format(Locale.US, "%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%.4f,%.2f,%.4f,%.2f,%.2f,%d",
                    botX, botY, gunHeading, enemyX, enemyY, distance, enemyHeading,
                    enemyVelocity, enemyAngle, firePower, bulletSpeed, hit);
        }
    }
}
